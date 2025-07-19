import json
import sys
from typing import Dict, List, Any
from ortools.sat.python import cp_model


def load_config(path: str = "schedule-config.json") -> Dict[str, Any]:
    """Load configuration file and apply defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    settings = data.get("settings", {})
    default_teacher_imp = settings.get("defaultTeacherImportance", [1])[0]
    default_student_imp = settings.get("defaultStudentImportance", [0])[0]
    default_opt_slot = settings.get("defaultOptimalSlot", [0])[0]

    for teacher in data.get("teachers", []):
        teacher.setdefault("importance", default_teacher_imp)

    for student in data.get("students", []):
        student.setdefault("importance", default_student_imp)
        student.setdefault("group", 1)

    for subj in data.get("subjects", {}).values():
        subj.setdefault("optimalSlot", default_opt_slot)
        if "cabinets" not in subj:
            subj["cabinets"] = list(data.get("cabinets", {}))

    return data


def _init_schedule(days: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Prepare empty schedule structure."""
    schedule = {}
    for day in days:
        name = day["name"]
        schedule[name] = {slot: [] for slot in day["slots"]}
    return schedule


def build_model(cfg: Dict[str, Any]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Build schedule using CP-SAT optimisation."""
    days = cfg["days"]
    subjects = cfg["subjects"]
    teachers = cfg.get("teachers", [])
    students = cfg.get("students", [])
    cabinets = cfg.get("cabinets", {})

    teacher_names = [t["name"] for t in teachers]
    teacher_map = {t["name"]: set(t["subjects"]) for t in teachers}
    subject_teachers = {
        sid: [t for t in teacher_map if sid in teacher_map[t]] for sid in subjects
    }

    students_by_subject: Dict[str, List[str]] = {}
    student_size: Dict[str, int] = {}
    for stu in students:
        name = stu["name"]
        student_size[name] = int(stu.get("group", 1))
        for sid in stu["subjects"]:
            students_by_subject.setdefault(sid, []).append(name)

    model = cp_model.CpModel()
    penalties = cfg.get("penalties", {})
    penalty_val = penalties.get("unoptimalSlot", [0])[0]
    gap_teacher_val = penalties.get("gapTeacher", [0])[0]
    gap_student_val = penalties.get("gapStudent", [0])[0]
    settings = cfg.get("settings", {})
    stud_weight = settings.get("studentsPenaltyWeight", [1])[0]
    teach_weight = settings.get("teachersPenaltyWeight", [1])[0]
    max_teacher_slots = settings.get("maxTeacherSlots", [0])[0]
    max_student_slots = settings.get("maxStudentSlots", [0])[0]
    default_student_imp = settings.get("defaultStudentImportance", [0])[0]
    default_teacher_imp = settings.get("defaultTeacherImportance", [1])[0]
    student_importance = {
        s["name"]: s.get("importance", default_student_imp) for s in students
    }
    teacher_importance = {
        t["name"]: t.get("importance", default_teacher_imp) for t in teachers
    }

    # candidate variables for each subject class (day/start only)
    candidates: Dict[tuple, List[Dict[str, Any]]] = {}
    # chosen day index for each class
    class_day_idx: Dict[tuple, cp_model.IntVar] = {}
    # teacher and cabinet selections for every class
    class_teacher: Dict[tuple, cp_model.IntVar] = {}
    class_cabinet: Dict[tuple, cp_model.IntVar] = {}
    teacher_choice: Dict[tuple, cp_model.BoolVar] = {}
    cabinet_choice: Dict[tuple, cp_model.BoolVar] = {}
    # helper mapping for allowed teacher/cabinet choices
    teacher_index = {name: i for i, name in enumerate(teacher_names)}
    cabinet_index = {name: i for i, name in enumerate(cabinets)}

    for sid, subj in subjects.items():
        allowed_teachers = subject_teachers.get(sid)
        if not allowed_teachers:
            raise ValueError(f"No teacher available for subject {sid}")
        allowed_cabinets = subj.get("cabinets", list(cabinets))
        class_lengths = subj["classes"]
        enrolled = students_by_subject.get(sid, [])
        class_size = sum(student_size[s] for s in enrolled)

        for idx, length in enumerate(class_lengths):
            key = (sid, idx)
            cand_list = []
            for day_idx, day in enumerate(days):
                dname = day["name"]
                slots = day["slots"]
                for start in slots:
                    if start + length - 1 > slots[-1]:
                        continue
                    var = model.NewBoolVar(f"x_{sid}_{idx}_{dname}_{start}")
                    diff = abs(start - subj.get("optimalSlot", 0))
                    stud_pen = sum(
                        student_importance[s] * student_size[s]
                        for s in enrolled
                    )
                    cand_list.append(
                        {
                            "var": var,
                            "day": dname,
                            "day_idx": day_idx,
                            "start": start,
                            "length": length,
                            "size": class_size,
                            "students": enrolled,
                            "penalty": diff * penalty_val * stud_pen * stud_weight,
                        }
                    )
            if not cand_list:
                raise RuntimeError(f"No slot for subject {sid} class {idx}")
            model.Add(sum(c["var"] for c in cand_list) == 1)
            day_var = model.NewIntVar(0, len(days) - 1, f"day_idx_{sid}_{idx}")
            model.Add(day_var == sum(c["day_idx"] * c["var"] for c in cand_list))
            class_day_idx[key] = day_var
            teacher_domain = [teacher_index[t] for t in allowed_teachers]
            class_teacher[key] = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(teacher_domain),
                f"teacher_{sid}_{idx}",
            )
            # helper booleans for chosen teacher
            for t in allowed_teachers:
                tv = model.NewBoolVar(f"is_{sid}_{idx}_teacher_{t}")
                model.Add(class_teacher[key] == teacher_index[t]).OnlyEnforceIf(tv)
                model.Add(class_teacher[key] != teacher_index[t]).OnlyEnforceIf(tv.Not())
                teacher_choice[(sid, idx, t)] = tv
            cab_domain = [
                cabinet_index[c]
                for c in allowed_cabinets
                if cabinets[c]["capacity"] >= class_size
            ]
            if not cab_domain:
                raise RuntimeError(f"No cabinet for subject {sid} class {idx}")
            class_cabinet[key] = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(cab_domain),
                f"cab_{sid}_{idx}",
            )
            for c in allowed_cabinets:
                if cabinets[c]["capacity"] < class_size:
                    continue
                cv = model.NewBoolVar(f"is_{sid}_{idx}_cab_{c}")
                model.Add(class_cabinet[key] == cabinet_index[c]).OnlyEnforceIf(cv)
                model.Add(class_cabinet[key] != cabinet_index[c]).OnlyEnforceIf(cv.Not())
                cabinet_choice[(sid, idx, c)] = cv
            candidates[key] = cand_list

    # at most one class of same subject per day
    for sid, subj in subjects.items():
        class_count = len(subj["classes"])
        if class_count <= 1:
            continue
        for day in days:
            vars_in_day = []
            for idx in range(class_count):
                vars_in_day.extend(
                    c["var"]
                    for c in candidates[(sid, idx)]
                    if c["day"] == day["name"]
                )
            if vars_in_day:
                model.Add(sum(vars_in_day) <= 1)

        # classes must appear in chronological order across days
        for idx in range(1, class_count):
            prev_k = (sid, idx - 1)
            curr_k = (sid, idx)
            model.Add(class_day_idx[curr_k] > class_day_idx[prev_k])

    # teacher/student/cabinet conflicts
    for day in days:
        dname = day["name"]
        for slot in day["slots"]:
            for teacher in teacher_names:
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    if teacher not in subject_teachers[sid]:
                        continue
                    tv = teacher_choice[(sid, idx, teacher)]
                    for c in cand_list:
                        if c["day"] == dname and slot in range(c["start"], c["start"] + c["length"]):
                            b = model.NewBoolVar(
                                f"teach_{sid}_{idx}_{teacher}_{dname}_{slot}_{c['start']}"
                            )
                            model.Add(b <= tv)
                            model.Add(b <= c["var"])
                            model.Add(b >= tv + c["var"] - 1)
                            involved.append(b)
                if involved:
                    model.Add(sum(involved) <= 1)

            for cab in cabinets:
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    if (sid, idx, cab) not in cabinet_choice:
                        continue
                    cv = cabinet_choice[(sid, idx, cab)]
                    for c in cand_list:
                        if c["day"] == dname and slot in range(c["start"], c["start"] + c["length"]):
                            b = model.NewBoolVar(
                                f"cab_{sid}_{idx}_{cab}_{dname}_{slot}_{c['start']}"
                            )
                            model.Add(b <= cv)
                            model.Add(b <= c["var"])
                            model.Add(b >= cv + c["var"] - 1)
                            involved.append(b)
                if involved:
                    model.Add(sum(involved) <= 1)

            for stu in students:
                sname = stu["name"]
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    if sname not in students_by_subject.get(sid, []):
                        continue
                    for c in cand_list:
                        if c["day"] == dname and slot in range(
                            c["start"], c["start"] + c["length"]
                        ):
                            involved.append(c["var"])
                if involved:
                    model.Add(sum(involved) <= 1)

    # build slot variables for teachers and students
    teacher_slot = {}
    student_slot = {}
    for day in days:
        dname = day["name"]
        for slot in day["slots"]:
            for t in teacher_names:
                var = model.NewBoolVar(f"teach_{t}_{dname}_{slot}")
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    if t not in subject_teachers[sid]:
                        continue
                    tv = teacher_choice[(sid, idx, t)]
                    for c in cand_list:
                        if c["day"] == dname and slot in range(c["start"], c["start"] + c["length"]):
                            b = model.NewBoolVar(
                                f"tvar_{sid}_{idx}_{t}_{dname}_{slot}_{c['start']}"
                            )
                            model.Add(b <= tv)
                            model.Add(b <= c["var"])
                            model.Add(b >= tv + c["var"] - 1)
                            involved.append(b)
                if involved:
                    model.AddMaxEquality(var, involved)
                else:
                    model.Add(var == 0)
                teacher_slot[(t, dname, slot)] = var

            for stu in students:
                sname = stu["name"]
                var = model.NewBoolVar(f"stud_{sname}_{dname}_{slot}")
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    if sname not in students_by_subject.get(sid, []):
                        continue
                    for c in cand_list:
                        if c["day"] == dname and slot in range(
                            c["start"], c["start"] + c["length"]
                        ):
                            involved.append(c["var"])
                if involved:
                    model.AddMaxEquality(var, involved)
                else:
                    model.Add(var == 0)
                student_slot[(sname, dname, slot)] = var

    # prefix and suffix to detect gaps
    teacher_gap_vars = []
    for t in teacher_names:
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            prefix = {}
            prev = None
            for s in slots:
                curr = teacher_slot[(t, dname, s)]
                if prev is None:
                    prefix[s] = curr
                else:
                    pv = model.NewBoolVar(f"pref_t_{t}_{dname}_{s}")
                    model.Add(pv >= prev)
                    model.Add(pv >= curr)
                    model.Add(pv <= prev + curr)
                    prefix[s] = pv
                prev = prefix[s]
            suffix = {}
            nxt = None
            for s in reversed(slots):
                curr = teacher_slot[(t, dname, s)]
                if nxt is None:
                    suffix[s] = curr
                else:
                    sv = model.NewBoolVar(f"suff_t_{t}_{dname}_{s}")
                    model.Add(sv >= nxt)
                    model.Add(sv >= curr)
                    model.Add(sv <= nxt + curr)
                    suffix[s] = sv
                nxt = suffix[s]
            for idx in range(1, len(slots) - 1):
                s = slots[idx]
                g = model.NewBoolVar(f"gap_t_{t}_{dname}_{s}")
                prevp = prefix[slots[idx - 1]]
                nextp = suffix[slots[idx + 1]]
                cur = teacher_slot[(t, dname, s)]
                model.Add(g <= prevp)
                model.Add(g <= nextp)
                model.Add(g + cur <= 1)
                model.Add(g >= prevp + nextp - cur - 1)
                teacher_gap_vars.append((g, t))
            if max_teacher_slots > 0:
                win = max_teacher_slots + 1
                for start in range(len(slots) - win + 1):
                    model.Add(
                        sum(
                            teacher_slot[(t, dname, slots[k])] for k in range(start, start + win)
                        )
                        <= max_teacher_slots
                    )

    student_gap_vars = []
    for stu in students:
        sname = stu["name"]
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            prefix = {}
            prev = None
            for s in slots:
                curr = student_slot[(sname, dname, s)]
                if prev is None:
                    prefix[s] = curr
                else:
                    pv = model.NewBoolVar(f"pref_s_{sname}_{dname}_{s}")
                    model.Add(pv >= prev)
                    model.Add(pv >= curr)
                    model.Add(pv <= prev + curr)
                    prefix[s] = pv
                prev = prefix[s]
            suffix = {}
            nxt = None
            for s in reversed(slots):
                curr = student_slot[(sname, dname, s)]
                if nxt is None:
                    suffix[s] = curr
                else:
                    sv = model.NewBoolVar(f"suff_s_{sname}_{dname}_{s}")
                    model.Add(sv >= nxt)
                    model.Add(sv >= curr)
                    model.Add(sv <= nxt + curr)
                    suffix[s] = sv
                nxt = suffix[s]
            for idx in range(1, len(slots) - 1):
                s = slots[idx]
                g = model.NewBoolVar(f"gap_s_{sname}_{dname}_{s}")
                prevp = prefix[slots[idx - 1]]
                nextp = suffix[slots[idx + 1]]
                cur = student_slot[(sname, dname, s)]
                model.Add(g <= prevp)
                model.Add(g <= nextp)
                model.Add(g + cur <= 1)
                model.Add(g >= prevp + nextp - cur - 1)
                student_gap_vars.append((g, sname))
            if max_student_slots > 0:
                win = max_student_slots + 1
                for start in range(len(slots) - win + 1):
                    model.Add(
                        sum(
                            student_slot[(sname, dname, slots[k])] for k in range(start, start + win)
                        )
                        <= max_student_slots
                    )

    # objective
    gap_teacher_expr = sum(
        gap_teacher_val * teacher_importance[t] * teach_weight * var for var, t in teacher_gap_vars
    )
    gap_student_expr = sum(
        gap_student_val
        * student_importance[s] * student_size[s] * stud_weight * var
        for var, s in student_gap_vars
    )
    base_obj = sum(
        c["penalty"] * c["var"] for cand_list in candidates.values() for c in cand_list
    )
    model.Minimize(base_obj + gap_teacher_expr + gap_student_expr)

    solver = cp_model.CpSolver()

    model_params = cfg.get("model", {})
    solver.parameters.max_time_in_seconds = model_params.get("maxTime", 300)
    solver.parameters.num_search_workers = model_params.get("workers", 10)
    solver.parameters.log_search_progress = model_params.get("showProgress", False)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for (sid, idx), cand_list in candidates.items():
        teach_idx = solver.Value(class_teacher[(sid, idx)])
        cab_idx = solver.Value(class_cabinet[(sid, idx)])
        teacher_name = teacher_names[teach_idx]
        cabinet_name = list(cabinets.keys())[cab_idx]
        for c in cand_list:
            if solver.Value(c["var"]):
                for s in range(c["start"], c["start"] + c["length"]):
                    schedule[c["day"]][s].append(
                        {
                            "subject": sid,
                            "teacher": teacher_name,
                            "cabinet": cabinet_name,
                            "students": c["students"],
                            "size": c["size"],
                            "start": c["start"],
                            "length": c["length"],
                        }
                    )

    return schedule


def solve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate schedule and wrap it in export format."""
    schedule = build_model(cfg)

    teacher_names = [t["name"] for t in cfg.get("teachers", [])]
    student_names = [s["name"] for s in cfg.get("students", [])]
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}

    teacher_slots = {
        t: {day["name"]: set() for day in cfg["days"]} for t in teacher_names
    }
    student_slots = {
        s: {day["name"]: set() for day in cfg["days"]} for s in student_names
    }

    for day in cfg["days"]:
        name = day["name"]
        for slot in day["slots"]:
            for cls in schedule[name][slot]:
                teacher_slots[cls["teacher"]][name].add(slot)
                for stu in cls["students"]:
                    student_slots[stu][name].add(slot)

    teacher_state = {
        t: {day["name"]: {} for day in cfg["days"]} for t in teacher_names
    }
    for t in teacher_names:
        for day in cfg["days"]:
            name = day["name"]
            slots = teacher_slots[t][name]
            if not slots:
                for s in day["slots"]:
                    teacher_state[t][name][s] = "home"
                continue
            first, last = min(slots), max(slots)
            for s in day["slots"]:
                if s < first or s > last:
                    teacher_state[t][name][s] = "home"
                elif s in slots:
                    teacher_state[t][name][s] = "class"
                else:
                    teacher_state[t][name][s] = "gap"

    student_state = {
        s: {day["name"]: {} for day in cfg["days"]} for s in student_names
    }
    for st in student_names:
        for day in cfg["days"]:
            name = day["name"]
            slots = student_slots[st][name]
            if not slots:
                for s in day["slots"]:
                    student_state[st][name][s] = "home"
                continue
            last = max(slots)
            for s in day["slots"]:
                if s > last:
                    student_state[st][name][s] = "home"
                elif s in slots:
                    student_state[st][name][s] = "class"
                else:
                    student_state[st][name][s] = "gap"

    settings = cfg.get("settings", {})
    penalties_cfg = {k: v[0] for k, v in cfg.get("penalties", {}).items()}
    teachers_w = settings.get("teachersPenaltyWeight", [1])[0]
    students_w = settings.get("studentsPenaltyWeight", [1])[0]
    def_teacher_imp = settings.get("defaultTeacherImportance", [1])[0]
    def_student_imp = settings.get("defaultStudentImportance", [0])[0]
    teacher_importance = {
        t["name"]: t.get("importance", def_teacher_imp)
        for t in cfg.get("teachers", [])
    }
    student_importance = {
        s["name"]: s.get("importance", def_student_imp)
        for s in cfg.get("students", [])
    }
    default_opt = settings.get("defaultOptimalSlot", [0])[0]

    slot_penalties = {
        day["name"]: {slot: {k: 0 for k in penalties_cfg} for slot in day["slots"]}
        for day in cfg["days"]
    }

    # calculate penalties per slot using individual importance
    for day in cfg["days"]:
        dname = day["name"]
        for slot in day["slots"]:
            # gap penalties for teachers
            for t in teacher_names:
                if teacher_state[t][dname][slot] == "gap":
                    p = penalties_cfg.get("gapTeacher", 0) * teacher_importance[t] * teachers_w
                    slot_penalties[dname][slot]["gapTeacher"] += p
            # gap penalties for students
            for sname in student_names:
                if student_state[sname][dname][slot] == "gap":
                    p = (
                        penalties_cfg.get("gapStudent", 0)
                        * student_importance[sname]
                        * student_size.get(sname, 1)
                        * students_w
                    )
                    slot_penalties[dname][slot]["gapStudent"] += p

    # penalties for unoptimal slots, assigned at class start
    for day in cfg["days"]:
        dname = day["name"]
        for slot in day["slots"]:
            for cls in schedule[dname][slot]:
                if slot != cls["start"]:
                    continue
                sid = cls["subject"]
                opt = cfg["subjects"][sid].get("optimalSlot", default_opt)
                diff = abs(cls["start"] - opt)
                if diff == 0:
                    continue
                base = penalties_cfg.get("unoptimalSlot", 0) * diff
                for sname in cls["students"]:
                    p = (
                        base
                        * student_importance[sname]
                        * student_size.get(sname, 1)
                        * students_w
                    )
                    slot_penalties[dname][slot]["unoptimalSlot"] += p

    total_penalty = 0
    for day_map in slot_penalties.values():
        for pen in day_map.values():
            total_penalty += sum(pen.values())

    export = {"days": []}
    for day in cfg["days"]:
        name = day["name"]
        slot_list = []
        for slot in day["slots"]:
            classes = schedule[name][slot]
            gaps_students = [s for s in student_names if student_state[s][name][slot] == "gap"]
            gaps_teachers = [t for t in teacher_names if teacher_state[t][name][slot] == "gap"]
            home_students = [s for s in student_names if student_state[s][name][slot] == "home"]
            home_teachers = [t for t in teacher_names if teacher_state[t][name][slot] == "home"]
            slot_list.append({
                "slotIndex": slot,
                "classes": classes,
                "gaps": {"students": gaps_students, "teachers": gaps_teachers},
                "home": {"students": home_students, "teachers": home_teachers},
                "penalty": slot_penalties[name][slot],
            })
        export["days"].append({"name": name, "slots": slot_list})
    export["totalPenalty"] = total_penalty

    return export


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "schedule-config.json"
    cfg = load_config(cfg_file)
    result = solve(cfg)
    with open("schedule.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("Schedule written to schedule.json")


if __name__ == "__main__":
    main()
