import json
import os
import sys
from collections import defaultdict
from statistics import mean
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
    default_teacher_arr = settings.get("defaultTeacherArriveEarly", [False])[0]
    default_student_arr = settings.get("defaultStudentArriveEarly", [True])[0]

    for teacher in data.get("teachers", []):
        teacher.setdefault("importance", default_teacher_imp)
        teacher.setdefault("arriveEarly", default_teacher_arr)

    for student in data.get("students", []):
        student.setdefault("importance", default_student_imp)
        student.setdefault("group", 1)
        student.setdefault("arriveEarly", default_student_arr)

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
    teacher_arrive = {t["name"]: bool(t.get("arriveEarly", False)) for t in teachers}
    subject_teachers = {
        sid: [t for t in teacher_map if sid in teacher_map[t]] for sid in subjects
    }

    students_by_subject: Dict[str, List[str]] = {}
    student_size: Dict[str, int] = {}
    student_arrive = {s["name"]: bool(s.get("arriveEarly", True)) for s in students}
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
        arrive = teacher_arrive.get(t, False)
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            prefix = {}
            prev = 1 if arrive else 0
            for s in slots:
                curr = teacher_slot[(t, dname, s)]
                pv = model.NewBoolVar(f"pref_t_{t}_{dname}_{s}")
                model.Add(pv >= prev)
                model.Add(pv >= curr)
                model.Add(pv <= prev + curr)
                prefix[s] = pv
                prev = pv
            suffix = {}
            nxt = 0
            for s in reversed(slots):
                curr = teacher_slot[(t, dname, s)]
                sv = model.NewBoolVar(f"suff_t_{t}_{dname}_{s}")
                model.Add(sv >= nxt)
                model.Add(sv >= curr)
                model.Add(sv <= nxt + curr)
                suffix[s] = sv
                nxt = sv
            for s in slots:
                g = model.NewBoolVar(f"gap_t_{t}_{dname}_{s}")
                cur = teacher_slot[(t, dname, s)]
                p = prefix[s]
                n = suffix[s]
                model.Add(g <= p)
                model.Add(g <= n)
                model.Add(g + cur <= 1)
                model.Add(g >= p + n - cur - 1)
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
        arrive = student_arrive.get(sname, True)
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            prefix = {}
            prev = 1 if arrive else 0
            for s in slots:
                curr = student_slot[(sname, dname, s)]
                pv = model.NewBoolVar(f"pref_s_{sname}_{dname}_{s}")
                model.Add(pv >= prev)
                model.Add(pv >= curr)
                model.Add(pv <= prev + curr)
                prefix[s] = pv
                prev = pv
            suffix = {}
            nxt = 0
            for s in reversed(slots):
                curr = student_slot[(sname, dname, s)]
                sv = model.NewBoolVar(f"suff_s_{sname}_{dname}_{s}")
                model.Add(sv >= nxt)
                model.Add(sv >= curr)
                model.Add(sv <= nxt + curr)
                suffix[s] = sv
                nxt = sv
            for s in slots:
                g = model.NewBoolVar(f"gap_s_{sname}_{dname}_{s}")
                cur = student_slot[(sname, dname, s)]
                p = prefix[s]
                n = suffix[s]
                model.Add(g <= p)
                model.Add(g <= n)
                model.Add(g + cur <= 1)
                model.Add(g >= p + n - cur - 1)
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

    # configuration parameters referenced later in the function
    settings = cfg.get("settings", {})

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
    teach_arrive = {
        t["name"]: t.get(
            "arriveEarly",
            settings.get("defaultTeacherArriveEarly", [False])[0],
        )
        for t in cfg.get("teachers", [])
    }

    for t in teacher_names:
        for day in cfg["days"]:
            name = day["name"]
            slots = teacher_slots[t][name]
            if not slots:
                for s in day["slots"]:
                    teacher_state[t][name][s] = "home"
                continue
            start = day["slots"][0] if teach_arrive.get(t, False) else min(slots)
            last = max(slots)
            for s in day["slots"]:
                if s < start or s > last:
                    teacher_state[t][name][s] = "home"
                elif s in slots:
                    teacher_state[t][name][s] = "class"
                else:
                    teacher_state[t][name][s] = "gap"

    student_state = {
        s: {day["name"]: {} for day in cfg["days"]} for s in student_names
    }
    stud_arrive = {s["name"]: s.get("arriveEarly", settings.get("defaultStudentArriveEarly", [True])[0]) for s in cfg.get("students", [])}
    for st in student_names:
        for day in cfg["days"]:
            name = day["name"]
            slots = student_slots[st][name]
            if not slots:
                for s in day["slots"]:
                    student_state[st][name][s] = "home"
                continue
            start = day["slots"][0] if stud_arrive.get(st, True) else min(slots)
            last = max(slots)
            for s in day["slots"]:
                if s < start or s > last:
                    student_state[st][name][s] = "home"
                elif s in slots:
                    student_state[st][name][s] = "class"
                else:
                    student_state[st][name][s] = "gap"

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
    slot_penalty_details = {
        day["name"]: {slot: [] for slot in day["slots"]}
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
                    slot_penalty_details[dname][slot].append({"name": t, "type": "gapTeacher", "amount": p})
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
                    slot_penalty_details[dname][slot].append({"name": sname, "type": "gapStudent", "amount": p})

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
                    slot_penalty_details[dname][slot].append({"name": sname, "type": "unoptimalSlot", "amount": p})

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
                "penaltyDetails": slot_penalty_details[name][slot],
            })
        export["days"].append({"name": name, "slots": slot_list})
    export["totalPenalty"] = total_penalty

    return export


def analyse_teachers(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Gather statistics about teacher workload."""
    teachers = defaultdict(
        lambda: {"blocks": 0, "students": [], "subjects": defaultdict(list)}
    )
    for day in schedule.get("days", []):
        for slot in day.get("slots", []):
            for cls in slot.get("classes", []):
                tid = cls["teacher"]
                sid = cls["subject"]
                count = cls.get("size", len(cls.get("students", [])))
                teachers[tid]["blocks"] += 1
                teachers[tid]["students"].append(count)
                teachers[tid]["subjects"][sid].append(count)
    return teachers


def analyse_students(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate hours per subject for every student."""
    student_hours = defaultdict(lambda: defaultdict(int))
    for day in schedule.get("days", []):
        for slot in day.get("slots", []):
            for cls in slot.get("classes", []):
                sid = cls["subject"]
                for st in cls.get("students", []):
                    student_hours[st][sid] += 1
    return student_hours


def analyse_subjects(schedule: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Collect information about subjects."""
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}
    subjects = defaultdict(
        lambda: {"students": set(), "student_count": 0, "teachers": set(), "class_sizes": []}
    )
    for day in schedule.get("days", []):
        for slot in day.get("slots", []):
            for cls in slot.get("classes", []):
                sid = cls["subject"]
                subjects[sid]["teachers"].add(cls["teacher"])
                subjects[sid]["class_sizes"].append(
                    cls.get("size", len(cls.get("students", [])))
                )
                for stu in cls.get("students", []):
                    if stu not in subjects[sid]["students"]:
                        subjects[sid]["students"].add(stu)
                        subjects[sid]["student_count"] += student_size.get(stu, 1)
    return subjects


def _split_two_parts(name: str, max_len: int = 12) -> tuple[str, str]:
    """Split long names for better table formatting."""
    if len(name) <= max_len:
        return name, ""
    half = len(name) // 2
    left = name.rfind(" ", 0, half)
    right = name.find(" ", half)
    if left == -1 and right == -1:
        return name[:half], name[half:]
    if left == -1:
        idx = right
    elif right == -1:
        idx = left
    else:
        idx = left if half - left <= right - half else right
    return name[:idx], name[idx + 1:]


def _format_table(rows, header_top=None, header_bottom=None, center_mask=None, join_rows=False):
    """Return simple ASCII table formatted as text."""
    col_count = max(len(r) for r in rows) if rows else 0
    if header_top:
        col_count = max(col_count, len(header_top))
    if header_bottom:
        col_count = max(col_count, len(header_bottom))

    rows = [list(r) + [""] * (col_count - len(r)) for r in rows]
    if header_top:
        header_top = list(header_top) + [""] * (col_count - len(header_top))
    if header_bottom:
        header_bottom = list(header_bottom) + [""] * (col_count - len(header_bottom))

    if center_mask is None:
        center_mask = [False] * col_count
    else:
        center_mask = list(center_mask) + [False] * (col_count - len(center_mask))

    widths = [0] * col_count
    sources = rows[:]
    if header_top:
        sources.append(header_top)
    if header_bottom:
        sources.append(header_bottom)
    for r in sources:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        parts = []
        for i in range(col_count):
            text = str(cells[i])
            if center_mask[i]:
                parts.append(text.center(widths[i]))
            else:
                parts.append(text.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    if header_top:
        lines.append(fmt_row(header_top))
        if header_bottom:
            lines.append(fmt_row(header_bottom))
        lines.append(sep)
    for idx, row in enumerate(rows):
        lines.append(fmt_row(row))
        if join_rows and idx % 2 == 0 and idx + 1 < len(rows):
            continue
        lines.append(sep)
    return "\n".join(lines)


def _teacher_table(teachers, teacher_names, subject_names):
    rows = []
    max_subj = max((len(info["subjects"]) for info in teachers.values()), default=0)

    def teacher_hours(info):
        return sum(len(c) for c in info["subjects"].values())

    for tid, info in sorted(teachers.items(), key=lambda x: teacher_hours(x[1]), reverse=True):
        name1, name2 = _split_two_parts(teacher_names.get(tid, tid))
        total_hours = teacher_hours(info)
        row_top = [name1]
        row_bottom = [name2]
        subj_stats = sorted(info["subjects"].items(), key=lambda x: len(x[1]), reverse=True)
        for sid, counts in subj_stats:
            row_top.append(subject_names.get(sid, sid))
            avg_s = mean(counts) if counts else 0
            row_bottom.append(f"{len(counts)} | {avg_s:.1f}")
        row_top += [""] * (max_subj - len(subj_stats))
        row_bottom += [""] * (max_subj - len(subj_stats))
        row_top.append("")
        row_bottom.append(str(total_hours))
        rows.append(row_top)
        rows.append(row_bottom)

    header_top = ["Teacher"] + [f"Subject #{i+1}" for i in range(max_subj)] + ["Total"]
    header_bottom = [""] + ["Classes | Avg" for _ in range(max_subj)] + ["Hours"]
    center_mask = [False] + [True] * max_subj + [True]
    return _format_table(rows, header_top, header_bottom, center_mask, join_rows=True)


def _student_list(students, student_names, subject_names):
    lines = []
    for sid, subj_map in sorted(students.items(), key=lambda x: sum(x[1].values()), reverse=True):
        name = student_names.get(sid, sid)
        subj_count = len(subj_map)
        total_hours = sum(subj_map.values())
        parts = []
        for sub_id, hours in sorted(subj_map.items(), key=lambda x: subject_names.get(x[0], x[0])):
            parts.append(f"{hours} hours {subject_names.get(sub_id, sub_id)}")
        line = f"{name} has {subj_count} subjects for {total_hours} hours: " + ", ".join(parts)
        lines.append(line)
    return "\n".join(lines)


def _subject_table(subjects, subject_names):
    rows = []
    for sid, info in sorted(subjects.items(), key=lambda x: subject_names.get(x[0], x[0])):
        name = subject_names.get(sid, sid)
        total_students = info.get("student_count", len(info.get("students", [])))
        teachers_cnt = len(info["teachers"])
        avg_size = mean(info["class_sizes"]) if info["class_sizes"] else 0
        rows.append([name, str(total_students), str(teachers_cnt), f"{avg_size:.1f}"])

    header = ["Subject", "Students", "Teachers", "Avg class"]
    center_mask = [False, True, True, True]
    return _format_table(rows, header, center_mask=center_mask)


def report_analysis(schedule: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    teachers = analyse_teachers(schedule)
    students = analyse_students(schedule)
    subjects = analyse_subjects(schedule, cfg)

    teacher_names = {t["name"]: t.get("name", t["name"]) for t in cfg.get("teachers", [])}
    student_names = {s["name"]: s.get("name", s["name"]) for s in cfg.get("students", [])}
    subject_names = {sid: info.get("name", sid) for sid, info in cfg.get("subjects", {}).items()}

    print("=== Teachers ===")
    print(_teacher_table(teachers, teacher_names, subject_names))

    print("\n=== Students ===")
    print(_student_list(students, student_names, subject_names))

    print("\n=== Subjects ===")
    print(_subject_table(subjects, subject_names))


def render_schedule(schedule: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Print schedule grouped by day and slot."""
    subject_names = {sid: info.get("name", sid) for sid, info in cfg.get("subjects", {}).items()}
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}

    print()
    for day in schedule.get("days", []):
        print(day.get("name", "Unknown"))
        for slot in day.get("slots", []):
            idx = slot.get("slotIndex")
            classes = slot.get("classes", [])
            gaps = slot.get("gaps", {})
            home = slot.get("home", {})
            gap_t = len(gaps.get("teachers", []))
            gap_s = sum(student_size.get(n, 1) for n in gaps.get("students", []))
            home_t = len(home.get("teachers", []))
            home_s = sum(student_size.get(n, 1) for n in home.get("students", []))

            header = f"  Slot {idx} [gap T:{gap_t} S:{gap_s} home T:{home_t} S:{home_s}]"
            if not classes:
                print(f"{header}: --")
                continue
            print(f"{header}:")
            for cls in classes:
                subj = subject_names.get(cls["subject"], cls["subject"])
                teacher = cls["teacher"]
                size = cls.get("size", len(cls.get("students", [])))
                length = cls.get("length", 1)
                start = cls.get("start", idx)
                part = ""
                if length > 1:
                    pno = idx - start + 1
                    part = f" (part {pno}/{length})"
                print(f"    {subj} by {teacher} for {size} students, {length}h{part}")
        print()


def generate_html(schedule: Dict[str, Any], cfg: Dict[str, Any], path: str = "schedule.html") -> None:
    """Create interactive HTML overview of the schedule."""
    schedule_json = json.dumps(schedule, ensure_ascii=False)
    cfg_json = json.dumps(cfg, ensure_ascii=False)
    html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Schedule</title>
<style>
body { font-family: Arial, sans-serif; }
.schedule-grid { border-collapse: collapse; width: 100%; display:grid; }
.schedule-grid .cell, .schedule-grid .header { border:1px solid #999; padding:4px; vertical-align:top; }
.schedule-grid .cell { display:flex; flex-direction:column; }
.schedule-grid .header { background:#f0f0f0; text-align:center; }
.class-block { display:flex; flex-direction:column; margin-bottom:4px; }
.class-line { display:flex; gap:4px; width:100%; }
.class-line span { flex:1; }
.cls-subj { flex:0 0 50%; text-align:left; }
.cls-room { flex:0 0 30%; text-align:right; }
.cls-part { flex:0 0 20%; text-align:right; }
.slot-info { display:flex; gap:4px; justify-content:space-between; font-size:0.9em; color:#555; cursor:pointer; margin-top:auto; }
.slot-info span { flex:1 1 20%; text-align:center; }
.modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); }
.modal-content { background:#fff; margin:5% auto; padding:20px; width:90%; max-height:90%; overflow:auto; }
.modal-header { position:relative; text-align:center; margin-bottom:10px; }
.close { position:absolute; right:0; top:0; cursor:pointer; font-size:20px; }
.nav { display:inline-flex; align-items:center; cursor:pointer; font-size:20px; margin:0 10px; }
.nav-lbl { font-size:0.8em; color:#888; margin:0 4px; }
.clickable { color:#0066cc; cursor:pointer; text-decoration:underline; }
.slot-detail .slot-class{border-top:1px solid #ddd;padding:2px 0;}
.slot-detail .slot-class:first-child{border-top:none;}
.detail-line{display:flex;gap:6px;}
.detail-line span{flex:1;}
.detail-subj{flex:0 0 40%;text-align:left;}
.detail-teacher{flex:0 0 20%;text-align:left;}
.detail-room{flex:0 0 15%;text-align:right;}
.detail-size{flex:0 0 10%;text-align:right;}
.detail-part{flex:0 0 15%;text-align:right;}
.detail-students{font-size:0.9em;margin-left:1em;}
.info-table{border-collapse:collapse;width:100%;margin-top:6px;}
.info-table th,.info-table td{border:1px solid #999;padding:4px;vertical-align:top;}
.info-table th{background:#f0f0f0;}
.info-table td.num{text-align:right;}
/* overview tables below the schedule */
.overview-section{margin-top:20px;}
.overview-table{border:1px solid #999;border-collapse:collapse;width:100%;}
.overview-header,.overview-row{display:flex;align-items:center;}
.overview-header span,.overview-row span{padding:4px;border-right:1px solid #999;flex:1;text-align:center;}
.overview-header span:last-child,.overview-row span:last-child{border-right:none;}
.overview-row{border-top:1px solid #999;}
.overview-row:first-child{border-top:none;}
.overview-header{background:#f0f0f0;font-weight:bold;}
.person-name{flex:0 0 14%;text-align:left;}
.person-info{flex:0 0 12%;}
.person-pen{flex:0 0 8%;text-align:right;}
.person-hours{flex:0 0 8%;text-align:right;}
.person-time{flex:0 0 8%;text-align:right;}
.person-subjects{flex:0 0 50%;text-align:left;padding:0;}
.subject-list{display:flex;flex-direction:column;}
.subject-line{display:flex;gap:6px;border-top:1px solid #ddd;padding:2px 4px;}
.subject-line:first-child{border-top:none;}
.subject-name{flex:0 0 60%;text-align:left;}
.subject-count{flex:0 0 20%;text-align:right;}
.subject-extra{flex:0 0 20%;text-align:right;}
</style>
</head>
<body>
<h1>Schedule Overview</h1>
<div id="table" class="schedule-grid"></div>
<h2 class="overview-section">Teachers</h2>
<div id="teachers" class="overview-table"></div>
<h2 class="overview-section">Students</h2>
<div id="students" class="overview-table"></div>
<div id="modal" class="modal"><div class="modal-content"><div class="modal-header"><span id="back" class="nav">&#9664;<span id="back-lbl" class="nav-lbl"></span></span><span id="forward" class="nav"><span id="fwd-lbl" class="nav-lbl"></span>&#9654;</span><span id="close" class="close">&#10006;</span></div><div id="modal-body"></div></div></div>
<script>
const scheduleData = __SCHEDULE__;
const configData = __CONFIG__;
</script>
<script>
(function(){
const modal=document.getElementById('modal');
const close=document.getElementById('close');
const backBtn=document.getElementById('back');
const fwdBtn=document.getElementById('forward');
const backLbl=document.getElementById('back-lbl');
const fwdLbl=document.getElementById('fwd-lbl');
const modalBody=document.getElementById('modal-body');
close.onclick=()=>{modal.style.display='none';};
backBtn.onclick=()=>{if(historyIndex>0){historyIndex--;renderModal();}};
fwdBtn.onclick=()=>{if(historyIndex<historyStack.length-1){historyIndex++;renderModal();}};
window.onclick=e=>{if(e.target==modal)modal.style.display='none';};
const studentSize={};
(configData.students||[]).forEach((s,i)=>{studentSize[s.name]=s.group||1;});
const teacherIndex={};
(configData.teachers||[]).forEach((t,i)=>{teacherIndex[t.name]=i;});
const studentIndex={};
(configData.students||[]).forEach((s,i)=>{studentIndex[s.name]=i;});
const teacherSet=new Set(Object.keys(teacherIndex));
const studentSet=new Set(Object.keys(studentIndex));
function personLink(name,role){
  if(role==='teacher' || (!role && teacherSet.has(name) && !studentSet.has(name))){
    const id=teacherIndex[name];
    return '<span class="clickable teacher" data-id="'+id+'">'+name+'</span>';
  }
  if(role==='student' || (!role && studentSet.has(name) && !teacherSet.has(name))){
    const id=studentIndex[name];
    return '<span class="clickable student" data-id="'+id+'">'+name+'</span>';
  }
  return name;
}
function countStudents(list){return(list||[]).reduce((a,n)=>a+(studentSize[n]||1),0);}
let historyStack=[];
let historyTitles=[];
let historyIndex=-1;
function getTitle(html){const m=html.match(/<h2[^>]*>(.*?)<\/h2>/i);return m?m[1]:'';}
function renderModal(){
 modalBody.innerHTML=historyStack[historyIndex]||'';
 modal.style.display='block';
 backBtn.style.display=historyIndex>0?'inline-flex':'none';
 fwdBtn.style.display=historyIndex<historyStack.length-1?'inline-flex':'none';
 backLbl.textContent=historyIndex>0?historyTitles[historyIndex-1]:'';
 fwdLbl.textContent=historyIndex<historyStack.length-1?historyTitles[historyIndex+1]:'';
}
function openModal(html,reset=true){
 const title=getTitle(html);
 if(reset){historyStack=[html];historyTitles=[title];historyIndex=0;}else{historyStack=historyStack.slice(0,historyIndex+1);historyTitles=historyTitles.slice(0,historyIndex+1);historyStack.push(html);historyTitles.push(title);historyIndex++;}
 renderModal();
}
const COLOR_MIN=[220,255,220];
const COLOR_MID=[255,255,255];
const COLOR_MAX=[255,220,220];

function buildTable(){
 const container=document.getElementById('table');
 container.style.gridTemplateColumns=`auto repeat(${scheduleData.days.length},1fr)`;
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 const cells=[];
 let minP=Infinity,maxP=-Infinity;
 container.innerHTML='';
 container.appendChild(document.createElement('div'));
 scheduleData.days.forEach(d=>{const h=document.createElement('div');h.className='header';h.textContent=d.name;container.appendChild(h);});
 for(let i=0;i<maxSlots;i++){
   const hdr=document.createElement('div');hdr.className='header';hdr.textContent='Slot '+i;container.appendChild(hdr);
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i) || {classes:[],gaps:{students:[],teachers:[]},home:{students:[],teachers:[]},penalty:{}};
     const cell=document.createElement('div');
     cell.className='cell';
     const pVal=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
     minP=Math.min(minP,pVal);maxP=Math.max(maxP,pVal);
     slot.classes.forEach(cls=>{
       const block=document.createElement('div');
       block.className='class-block';
       const subj=(configData.subjects[cls.subject]||{}).name||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const l1=document.createElement('div');
       l1.className='class-line';
       l1.innerHTML='<span class="cls-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
        '<span class="cls-room clickable cabinet" data-id="'+cls.cabinet+'">'+cls.cabinet+'</span>' +
        '<span class="cls-part">'+part+'</span>';
       const l2=document.createElement('div');
       l2.className='class-line';
       l2.innerHTML='<span class="cls-teach clickable teacher" data-id="'+teacherIndex[cls.teacher]+'">'+cls.teacher+'</span>'+
        '<span class="cls-size">'+cls.size+'</span>';
       block.appendChild(l1);block.appendChild(l2);
       cell.appendChild(block);
     });
     const info=document.createElement('div');
     info.className='slot-info';
     info.dataset.day=day.name;info.dataset.slot=i;
    function makeSpan(val,title){const s=document.createElement('span');s.textContent=val;s.title=title;return s;}
    const detail=(slot.penaltyDetails||[]).map(p=>p.name+' '+p.type+': '+p.amount.toFixed(1)).join('\n');
    info.appendChild(makeSpan(pVal.toFixed(1),detail||'Penalty'));
     info.appendChild(makeSpan(countStudents(slot.home.students),'Students at home: '+(slot.home.students.join(', ')||'-')));
     info.appendChild(makeSpan(slot.home.teachers.length,'Teachers at home: '+(slot.home.teachers.join(', ')||'-')));
     info.appendChild(makeSpan(countStudents(slot.gaps.students),'Students waiting for class: '+(slot.gaps.students.join(', ')||'-')));
     info.appendChild(makeSpan(slot.gaps.teachers.length,'Teachers waiting for class: '+(slot.gaps.teachers.join(', ')||'-')));
     cell.appendChild(info);
     container.appendChild(cell);
     cells.push({el:cell,val:pVal});
   });
 }

 const mid=(minP+maxP)/2;
 function mix(a,b,f){return a+(b-a)*f;}
 function colorFor(v){
   if(maxP===minP)return `rgb(${COLOR_MID.join(',')})`;
   if(v<=mid){
     const f=(v-minP)/(mid-minP||1);
     const r=Math.round(mix(COLOR_MIN[0],COLOR_MID[0],f));
     const g=Math.round(mix(COLOR_MIN[1],COLOR_MID[1],f));
     const b=Math.round(mix(COLOR_MIN[2],COLOR_MID[2],f));
     return `rgb(${r},${g},${b})`;
   }
   const f=(v-mid)/(maxP-mid||1);
   const r=Math.round(mix(COLOR_MID[0],COLOR_MAX[0],f));
   const g=Math.round(mix(COLOR_MID[1],COLOR_MAX[1],f));
   const b=Math.round(mix(COLOR_MID[2],COLOR_MAX[2],f));
   return `rgb(${r},${g},${b})`;
 }
 cells.forEach(c=>{c.el.style.background=colorFor(c.val);});
}

function showSlot(day,idx,fromModal=false){
 const d=scheduleData.days.find(x=>x.name===day);if(!d)return;
 const slot=d.slots.find(s=>s.slotIndex==idx);if(!slot)return;
 const total=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
 let html='<h2>'+day+' slot '+idx+'</h2><p>Total penalty: '+total.toFixed(1)+'</p>';
 html+='<div class="slot-detail">';
 slot.classes.forEach((cls)=>{
   const subj=(configData.subjects[cls.subject]||{}).name||cls.subject;
   const part=(cls.length>1)?((idx-cls.start+1)+'/'+cls.length):'1/1';
   html+='<div class="slot-class">'+
     '<div class="detail-line">'+
       '<span class="detail-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
       '<span class="detail-teacher clickable teacher" data-id="'+teacherIndex[cls.teacher]+'">'+cls.teacher+'</span>'+
       '<span class="detail-room clickable cabinet" data-id="'+cls.cabinet+'">'+cls.cabinet+'</span>'+
       '<span class="detail-size">'+cls.size+'</span>'+
       '<span class="detail-part">'+part+'</span>'+
     '</div>';
   const studs=cls.students.map(n=>personLink(n,'student')).join(', ');
   if(studs)html+='<div class="detail-students">'+studs+'</div>';
   html+='</div>';
 });
 html+='</div>';
 const homeStu=slot.home.students.map(n=>personLink(n,'student')).join(', ');
 const homeTeach=slot.home.teachers.map(n=>personLink(n,'teacher')).join(', ');
 const waitStu=slot.gaps.students.map(n=>personLink(n,'student')).join(', ');
 const waitTeach=slot.gaps.teachers.map(n=>personLink(n,'teacher')).join(', ');
 html+='<h3>Presence</h3>';
 html+='<table class="info-table"><tr><th></th><th>Students</th><th>Teachers</th></tr>'+
  '<tr><td>At home</td><td>'+ (homeStu||'-') +'</td><td>'+ (homeTeach||'-') +'</td></tr>'+
  '<tr><td>Waiting</td><td>'+ (waitStu||'-') +'</td><td>'+ (waitTeach||'-') +'</td></tr>'+
  '</table>';
 const penGrouped={};
 (slot.penaltyDetails||[]).filter(p=>p.amount>0).forEach(p=>{(penGrouped[p.type]=penGrouped[p.type]||[]).push(p);});
 const types=Object.keys(penGrouped);
 if(types.length){
   html+='<h3>Penalties</h3><table class="info-table"><tr><th>Type</th><th>Amount</th><th>Who</th></tr>';
   types.forEach(t=>{
     const list=penGrouped[t];
     const amount=list.reduce((a,x)=>a+x.amount,0);
     if(amount>0){
      const names=list.map(p=>{
        const role=p.type==='gapTeacher'?'teacher':'student';
        return personLink(p.name,role)+' ('+p.amount.toFixed(1)+')';
      }).join(', ');
       html+='<tr><td>'+t+'</td><td class="num">'+amount.toFixed(1)+'</td><td>'+names+'</td></tr>';
     }
   });
   html+='</table>';
 }
 openModal(html,!fromModal);
}

function computeTeacherStats(name){
 const info=(configData.teachers||[]).find(t=>t.name===name)||{};
 const defArr=(configData.settings.defaultTeacherArriveEarly||[false])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let sizes=[],total=0,gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>c.teacher===name));
   if(teachSlots.length){
     const firstClass=teachSlots[0].slotIndex;
     const first=arrive?dayStart:firstClass;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     teachSlots.forEach(sl=>{const c=sl.classes.find(x=>x.teacher===name);sizes.push(c.size);total++;});
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
   }
  });
 const avg=sizes.reduce((a,b)=>a+b,0)/(sizes.length||1);
 return{totalClasses:total,avgSize:avg.toFixed(1),gap:gap,time:time};
}

function computeStudentStats(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.settings.defaultStudentArriveEarly||[true])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>c.students.includes(name)));
   if(stSlots.length){
     const firstClass=stSlots[0].slotIndex;
     const first=arrive?dayStart:firstClass;
     const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
   }
 });
return{gap:gap,time:time};
}

function computeTeacherInfo(name){
 const info=(configData.teachers||[]).find(t=>t.name===name)||{};
 const defArr=(configData.settings.defaultTeacherArriveEarly||[false])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let hours=0,gap=0,time=0,subjects={},pen=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>c.teacher===name));
   if(teachSlots.length){
     const first=arrive?dayStart:teachSlots[0].slotIndex;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
     teachSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>c.teacher===name);
       hours++;
       const stat=subjects[cls.subject]||{count:0,size:0};
       stat.count++;stat.size+=cls.size;subjects[cls.subject]=stat;
     });
   }
  slots.forEach(sl=>{(sl.penaltyDetails||[]).forEach(p=>{if(p.name===name)pen+=p.amount;});});
 });
 for(const k in subjects){subjects[k].avg=(subjects[k].size/subjects[k].count).toFixed(1);}
 return{arrive,imp:info.importance,penalty:pen,hours:hours,time:time,subjects};
}

function computeStudentInfo(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.settings.defaultStudentArriveEarly||[true])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let hours=0,gap=0,time=0,subjects={},pen=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>c.students.includes(name)));
   if(stSlots.length){
     const first=arrive?dayStart:stSlots[0].slotIndex;
     const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
     stSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>c.students.includes(name));
       hours++;
       const stat=subjects[cls.subject]||{count:0,penalty:0};
       stat.count++;subjects[cls.subject]=stat;
     });
   }
   slots.forEach(sl=>{(sl.penaltyDetails||[]).forEach(p=>{
    if(p.name===name){
      pen+=p.amount;
      if(p.type==='unoptimalSlot'){
        const cls=sl.classes.find(c=>c.students.includes(name));
        if(cls){
          subjects[cls.subject]=subjects[cls.subject]||{count:0,penalty:0};
          subjects[cls.subject].penalty+=p.amount;
        }
      }
    }
  });});
});
return{arrive,imp:info.importance,penalty:pen,hours:hours,time:time,subjects};
}

function buildTeachers(){
 const cont=document.getElementById('teachers');
 cont.innerHTML='';
 const header=document.createElement('div');
 header.className='overview-header';
  header.innerHTML='<span class="person-name">Teacher</span><span class="person-info">Priority<br>Arrive</span><span class="person-pen">Penalty</span><span class="person-hours">Hours</span><span class="person-time">At school</span><span class="person-subjects">Subject<br>Cls | Avg</span>';
 cont.appendChild(header);
 const infos=(configData.teachers||[]).map(t=>{return{info:t,stat:computeTeacherInfo(t.name)}});
 infos.sort((a,b)=>b.stat.penalty-a.stat.penalty);
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
   const arr=item.stat.arrive?"yes":"no";
   const pr=item.info.importance!==undefined?item.info.importance:(configData.settings.defaultTeacherImportance||[1])[0];
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
     const name=(configData.subjects[sid]||{}).name||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+s.avg+'</span></div>';
   });
   row.innerHTML='<span class="person-name clickable teacher" data-id="'+teacherIndex[item.info.name]+'">'+item.info.name+'</span>'+
     '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
     '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>'+
     '<span class="person-hours">'+item.stat.hours+'</span>'+
     '<span class="person-time">'+item.stat.time+'</span>'+
     '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
   cont.appendChild(row);
  });
}

function buildStudents(){
 const cont=document.getElementById('students');
 cont.innerHTML='';
 const header=document.createElement('div');
 header.className='overview-header';
  header.innerHTML='<span class="person-name">Student</span><span class="person-info">Priority<br>Arrive</span><span class="person-pen">Penalty</span><span class="person-hours">Hours</span><span class="person-time">At school</span><span class="person-subjects">Subject<br>Cls | Pen</span>';
 cont.appendChild(header);
 const infos=(configData.students||[]).map(s=>{return{info:s,stat:computeStudentInfo(s.name)}});
 infos.sort((a,b)=>b.stat.penalty-a.stat.penalty);
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
   const arr=item.stat.arrive?"yes":"no";
   const pr=item.info.importance!==undefined?item.info.importance:(configData.settings.defaultStudentImportance||[0])[0];
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
     const name=(configData.subjects[sid]||{}).name||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+(s.penalty||0).toFixed(1)+'</span></div>';
   });
   row.innerHTML='<span class="person-name clickable student" data-id="'+studentIndex[item.info.name]+'">'+item.info.name+'</span>'+
     '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
     '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>'+
     '<span class="person-hours">'+item.stat.hours+'</span>'+
     '<span class="person-time">'+item.stat.time+'</span>'+
     '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
   cont.appendChild(row);
  });
}

function showTeacher(idx,fromModal=false){
 const info=(configData.teachers||[])[idx]||{};
 const name=info.name||'';
 const defImp=(configData.settings.defaultTeacherImportance||[1])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const stats=computeTeacherStats(name);
 let html='<h2>Teacher: '+name+'</h2><p>Importance: '+imp+'</p>';
 html+='<p>Total blocks: '+stats.totalClasses+', average class size: '+stats.avgSize+'</p>';
 html+='<p>Gap hours: '+stats.gap+', time at school: '+stats.time+'</p>';
 html+='<h3>Subjects</h3><ul>';
 (info.subjects||[]).forEach(sid=>{const sn=(configData.subjects[sid]||{}).name||sid;html+='<li><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></li>';});
 html+='</ul><h3>Schedule</h3><ul>';
scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.teacher===name){const sn=(configData.subjects[cls.subject]||{}).name||cls.subject;const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable subject" data-id="'+cls.subject+'">'+sn+'</span> ('+cls.size+' st)'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html,!fromModal);
}

function showStudent(idx,fromModal=false){
 const info=(configData.students||[])[idx]||{};
 const name=info.name||'';
 const defImp=(configData.settings.defaultStudentImportance||[0])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const stats=computeStudentStats(name);
 let html='<h2>Student: '+name+'</h2><p>Group size: '+(studentSize[name]||1)+'</p><p>Importance: '+imp+'</p>';
 html+='<p>Gap hours: '+stats.gap+', time at school: '+stats.time+'</p>';
 html+='<h3>Subjects</h3><ul>';
 (info.subjects||[]).forEach(sid=>{const sn=(configData.subjects[sid]||{}).name||sid;html+='<li><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></li>';});
 html+='</ul><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.students.includes(name)){const sn=(configData.subjects[cls.subject]||{}).name||cls.subject;const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable subject" data-id="'+cls.subject+'">'+sn+'</span> with <span class="clickable teacher" data-id="'+teacherIndex[cls.teacher]+'">'+cls.teacher+'</span>'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html,!fromModal);
}

function showCabinet(name,fromModal=false){
 const info=configData.cabinets[name]||{};
 let html='<h2>Room: '+name+'</h2><p>Capacity: '+(info.capacity||'-')+'</p><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.cabinet===name){const sn=(configData.subjects[cls.subject]||{}).name||cls.subject;const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable subject" data-id="'+cls.subject+'">'+sn+'</span> by <span class="clickable teacher" data-id="'+teacherIndex[cls.teacher]+'">'+cls.teacher+'</span> ('+cls.size+' st)'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html,!fromModal);
}

function showSubject(id,fromModal=false){
 const subj=configData.subjects[id]||{};
 const defOpt=(configData.settings.defaultOptimalSlot||[0])[0];
 let html='<h2>Subject: '+(subj.name||id)+'</h2>';
 html+='<p>Classes: '+(subj.classes||[]).join(', ')+'</p>';
 html+='<p>Optimal slot: '+(subj.optimalSlot!==undefined?subj.optimalSlot:defOpt)+'</p>';
 html+='<h3>Teachers</h3><ul>';
 (configData.teachers||[]).forEach((t,i)=>{if((t.subjects||[]).includes(id)){html+='<li><span class="clickable teacher" data-id="'+i+'">'+t.name+'</span></li>';}});
 html+='</ul><h3>Students</h3><ul>';
 (configData.students||[]).forEach((s,i)=>{if((s.subjects||[]).includes(id)){html+='<li><span class="clickable student" data-id="'+i+'">'+s.name+'</span> ('+(studentSize[s.name]||1)+')</li>';}});
 html+='</ul><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.subject===id){const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable teacher" data-id="'+teacherIndex[cls.teacher]+'">'+cls.teacher+'</span> ('+cls.size+' st)'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html,!fromModal);
}

document.addEventListener('click',e=>{
 const fromModal=modal.contains(e.target);
 const slotElem=e.target.closest('.slot-info');
 if(slotElem){showSlot(slotElem.dataset.day,parseInt(slotElem.dataset.slot),fromModal);return;}
 const t=e.target;
 if(t.classList.contains('subject')){showSubject(t.dataset.id,fromModal);}
 else if(t.classList.contains('teacher')){showTeacher(parseInt(t.dataset.id),fromModal);}
 else if(t.classList.contains('student')){showStudent(parseInt(t.dataset.id),fromModal);}
 else if(t.classList.contains('cabinet')){showCabinet(t.dataset.id,fromModal);}
});

buildTable();
buildTeachers();
buildStudents();
})();
</script>
</body>
</html>
"""
    html = html.replace("__SCHEDULE__", schedule_json).replace("__CONFIG__", cfg_json)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-y"]
    auto_yes = "-y" in sys.argv[1:]

    cfg_path = args[0] if len(args) >= 1 else "schedule-config.json"
    out_path = args[1] if len(args) >= 2 else "schedule.json"

    if not os.path.exists(cfg_path):
        print(f"Config '{cfg_path}' not found.")
        return

    skip_solve = False
    if os.path.exists(out_path):
        if auto_yes:
            skip_solve = True
        else:
            ans = input(f"Schedule file '{out_path}' found. Skip solving and use it? [y/N] ")
            skip_solve = ans.strip().lower().startswith("y")

    cfg = load_config(cfg_path)
    if skip_solve:
        with open(out_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
    else:
        result = solve(cfg)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"Schedule written to {out_path}")

    render_schedule(result, cfg)

    if auto_yes:
        show_analysis = True
    else:
        ans = input("Show analysis? [y/N] ")
        show_analysis = ans.strip().lower().startswith("y")

    if show_analysis:
        report_analysis(result, cfg)
        generate_html(result, cfg)
        print("schedule.html generated")


if __name__ == "__main__":
    main()
