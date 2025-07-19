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
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Schedule</title>
<style>
body { font-family: Arial, sans-serif; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #999; padding: 4px; vertical-align: top; }
th { background: #f0f0f0; }
.slot-info { font-size: 0.9em; color: #555; margin-top: 2px; }
.modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); }
.modal-content { background:#fff; margin:5% auto; padding:20px; width:90%; max-height:90%; overflow:auto; }
.close { float:right; cursor:pointer; font-size:20px; }
.clickable { color:#0066cc; cursor:pointer; text-decoration:underline; }
</style>
</head>
<body>
<h1>Schedule Overview</h1>
<div id="table"></div>
<div id="modal" class="modal"><div class="modal-content"><span id="close" class="close">&#10006;</span><div id="modal-body"></div></div></div>
<script>
const scheduleData = __SCHEDULE__;
const configData = __CONFIG__;
</script>
<script>
(function(){
const modal=document.getElementById('modal');
const close=document.getElementById('close');
const modalBody=document.getElementById('modal-body');
close.onclick=()=>{modal.style.display='none';};
window.onclick=e=>{if(e.target==modal)modal.style.display='none';};
const studentSize={};
(configData.students||[]).forEach(s=>{studentSize[s.name]=s.group||1;});
function countStudents(list){return(list||[]).reduce((a,n)=>a+(studentSize[n]||1),0);}
function openModal(html){modalBody.innerHTML=html;modal.style.display='block';}

function buildTable(){
 const container=document.getElementById('table');
 const table=document.createElement('table');
 const header=document.createElement('tr');
 header.appendChild(document.createElement('th'));
 scheduleData.days.forEach(d=>{const th=document.createElement('th');th.textContent=d.name;header.appendChild(th);});
 table.appendChild(header);
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 for(let i=0;i<maxSlots;i++){
   const tr=document.createElement('tr');
   const th=document.createElement('th');th.textContent='Slot '+i;tr.appendChild(th);
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i) || {classes:[],gaps:{students:[],teachers:[]},home:{students:[],teachers:[]}};
     const td=document.createElement('td');
     slot.classes.forEach(cls=>{
       const div=document.createElement('div');
       const subj=(configData.subjects[cls.subject]||{}).name||cls.subject;
       const part=cls.length>1?(' ('+(i-cls.start+1)+'/'+cls.length+')'):'';
       div.innerHTML='<span class="clickable subject" data-id="'+cls.subject+'">'+subj+'</span> by <span class="clickable teacher" data-id="'+cls.teacher+'">'+cls.teacher+'</span> ('+cls.size+' st, <span class="clickable cabinet" data-id="'+cls.cabinet+'">'+cls.cabinet+'</span>, '+cls.length+'h'+part+')';
       td.appendChild(div);
     });
     const info=document.createElement('div');
     info.className='slot-info clickable';
     info.dataset.day=day.name;info.dataset.slot=i;
     const gapT=slot.gaps.teachers.length;
     const gapS=countStudents(slot.gaps.students);
     const homeT=slot.home.teachers.length;
     const homeS=countStudents(slot.home.students);
     info.textContent='gap T'+gapT+' S'+gapS+' home T'+homeT+' S'+homeS;
     td.appendChild(info);
     tr.appendChild(td);
   });
   table.appendChild(tr);
 }
 container.appendChild(table);
}

function showSlot(day,idx){
 const d=scheduleData.days.find(x=>x.name===day);if(!d)return;
 const slot=d.slots.find(s=>s.slotIndex==idx);if(!slot)return;
 let html='<h2>'+day+' slot '+idx+'</h2>';
 slot.classes.forEach(cls=>{
   const subj=(configData.subjects[cls.subject]||{}).name||cls.subject;
   const part=cls.length>1?(' (part '+(idx-cls.start+1)+'/'+cls.length+')'):'';
   const studs=cls.students.map(n=>'<span class="clickable student" data-id="'+n+'">'+n+'</span>').join(', ');
   html+='<div><b>'+subj+'</b> by <span class="clickable teacher" data-id="'+cls.teacher+'">'+cls.teacher+'</span> for '+cls.size+' students in <span class="clickable cabinet" data-id="'+cls.cabinet+'">'+cls.cabinet+'</span> ('+cls.length+'h'+part+')<br>Students: '+studs+'</div>';
 });
 html+='<h3>Gaps</h3>Teachers: '+(slot.gaps.teachers.join(', ')||'-')+'<br>Students: '+((slot.gaps.students||[]).join(', ')||'-');
 html+='<h3>Home</h3>Teachers: '+(slot.home.teachers.join(', ')||'-')+'<br>Students: '+((slot.home.students||[]).join(', ')||'-');
 openModal(html);
}

function computeTeacherStats(name){
 let sizes=[],total=0,gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>c.teacher===name));
   if(teachSlots.length){
     const first=teachSlots[0].slotIndex;const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     teachSlots.forEach(sl=>{const c=sl.classes.find(x=>x.teacher===name);sizes.push(c.size);total++;});
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
   }
 });
 const avg=sizes.reduce((a,b)=>a+b,0)/(sizes.length||1);
 return{totalClasses:total,avgSize:avg.toFixed(1),gap:gap,time:time};
}

function computeStudentStats(name){
 let gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>c.students.includes(name)));
   if(stSlots.length){
     const first=stSlots[0].slotIndex;const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
   }
 });
 return{gap:gap,time:time};
}

function showTeacher(name){
 const info=(configData.teachers||[]).find(t=>t.name===name)||{};
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
 openModal(html);
}

function showStudent(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defImp=(configData.settings.defaultStudentImportance||[0])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const stats=computeStudentStats(name);
 let html='<h2>Student: '+name+'</h2><p>Group size: '+(studentSize[name]||1)+'</p><p>Importance: '+imp+'</p>';
 html+='<p>Gap hours: '+stats.gap+', time at school: '+stats.time+'</p>';
 html+='<h3>Subjects</h3><ul>';
 (info.subjects||[]).forEach(sid=>{const sn=(configData.subjects[sid]||{}).name||sid;html+='<li><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></li>';});
 html+='</ul><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.students.includes(name)){const sn=(configData.subjects[cls.subject]||{}).name||cls.subject;const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable subject" data-id="'+cls.subject+'">'+sn+'</span> with <span class="clickable teacher" data-id="'+cls.teacher+'">'+cls.teacher+'</span>'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html);
}

function showCabinet(name){
 const info=configData.cabinets[name]||{};
 let html='<h2>Room: '+name+'</h2><p>Capacity: '+(info.capacity||'-')+'</p><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.cabinet===name){const sn=(configData.subjects[cls.subject]||{}).name||cls.subject;const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable subject" data-id="'+cls.subject+'">'+sn+'</span> by <span class="clickable teacher" data-id="'+cls.teacher+'">'+cls.teacher+'</span> ('+cls.size+' st)'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html);
}

function showSubject(id){
 const subj=configData.subjects[id]||{};
 const defOpt=(configData.settings.defaultOptimalSlot||[0])[0];
 let html='<h2>Subject: '+(subj.name||id)+'</h2>';
 html+='<p>Classes: '+(subj.classes||[]).join(', ')+'</p>';
 html+='<p>Optimal slot: '+(subj.optimalSlot!==undefined?subj.optimalSlot:defOpt)+'</p>';
 html+='<h3>Teachers</h3><ul>';
 (configData.teachers||[]).forEach(t=>{if((t.subjects||[]).includes(id)){html+='<li><span class="clickable teacher" data-id="'+t.name+'">'+t.name+'</span></li>';}});
 html+='</ul><h3>Students</h3><ul>';
 (configData.students||[]).forEach(s=>{if((s.subjects||[]).includes(id)){html+='<li><span class="clickable student" data-id="'+s.name+'">'+s.name+'</span> ('+(studentSize[s.name]||1)+')</li>';}});
 html+='</ul><h3>Schedule</h3><ul>';
 scheduleData.days.forEach(day=>{day.slots.forEach(sl=>{sl.classes.forEach(cls=>{if(cls.subject===id){const part=cls.length>1?' (part '+(sl.slotIndex-cls.start+1)+'/'+cls.length+')':'';html+='<li>'+day.name+' slot '+sl.slotIndex+': <span class="clickable teacher" data-id="'+cls.teacher+'">'+cls.teacher+'</span> ('+cls.size+' st)'+part+'</li>';}});});});
 html+='</ul>';
 openModal(html);
}

document.addEventListener('click',e=>{
 const t=e.target;
 if(t.classList.contains('slot-info')){showSlot(t.dataset.day,parseInt(t.dataset.slot));}
 else if(t.classList.contains('subject')){showSubject(t.dataset.id);}
 else if(t.classList.contains('teacher')){showTeacher(t.dataset.id);}
 else if(t.classList.contains('student')){showStudent(t.dataset.id);}
 else if(t.classList.contains('cabinet')){showCabinet(t.dataset.id);}
});

buildTable();
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
