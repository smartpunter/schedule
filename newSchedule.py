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

    # map names to indices for compact integer variables
    teacher_index = {name: i for i, name in enumerate(teacher_names)}
    cabinet_index = {name: i for i, name in enumerate(cabinets)}
    day_index = {d["name"]: i for i, d in enumerate(days)}

    # offsets for building a continuous horizon for intervals
    offsets: List[int] = []
    offs = 0
    for d in days:
        offsets.append(offs)
        offs += len(d["slots"])
    horizon = offs

    # decision variables for each class
    class_vars = {}
    teacher_intervals: Dict[str, List[tuple]] = {n: [] for n in teacher_names}
    cabinet_intervals: Dict[str, List[tuple]] = {c: [] for c in cabinets}
    student_intervals: Dict[str, List[tuple]] = {s["name"]: [] for s in students}

    base_penalties = []

    for sid, subj in subjects.items():
        allowed_teachers = [teacher_index[t] for t in subject_teachers.get(sid, [])]
        if not allowed_teachers:
            raise ValueError(f"No teacher available for subject {sid}")
        allowed_cabinets = [cabinet_index[c] for c in subj.get("cabinets", list(cabinets))]
        class_lengths = subj["classes"]
        enrolled = students_by_subject.get(sid, [])
        class_size = sum(student_size[s] for s in enrolled)

        for idx, length in enumerate(class_lengths):
            day_var = model.NewIntVar(0, len(days) - 1, f"day_{sid}_{idx}")
            slot_var = model.NewIntVar(0, max(len(d["slots"]) for d in days) - 1, f"slot_{sid}_{idx}")
            allowed = []
            for d_idx, day in enumerate(days):
                for s in day["slots"]:
                    if s + length - 1 <= day["slots"][-1]:
                        allowed.append([d_idx, s])
            model.AddAllowedAssignments([day_var, slot_var], allowed)

            teach_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(allowed_teachers), f"teacher_{sid}_{idx}")
            cab_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(allowed_cabinets), f"cab_{sid}_{idx}")

            off_var = model.NewIntVar(0, horizon, f"off_{sid}_{idx}")
            model.AddElement(day_var, offsets, off_var)
            start_glob = model.NewIntVar(0, horizon, f"gstart_{sid}_{idx}")
            end_glob = model.NewIntVar(0, horizon, f"gend_{sid}_{idx}")
            model.Add(start_glob == slot_var + off_var)
            model.Add(end_glob == start_glob + length)

            interval = model.NewIntervalVar(start_glob, length, end_glob, f"int_{sid}_{idx}")

            # optional teacher intervals
            for t_idx in allowed_teachers:
                tname = teacher_names[t_idx]
                p = model.NewBoolVar(f"teach_{sid}_{idx}_{tname}")
                model.Add(teach_var == t_idx).OnlyEnforceIf(p)
                model.Add(teach_var != t_idx).OnlyEnforceIf(p.Not())
                t_int = model.NewOptionalIntervalVar(start_glob, length, end_glob, p, f"ti_{sid}_{idx}_{tname}")
                teacher_intervals[tname].append((t_int, day_var, length, p))

            # optional cabinet intervals
            for c_idx in allowed_cabinets:
                cname = list(cabinets)[c_idx]
                p = model.NewBoolVar(f"cab_{sid}_{idx}_{cname}")
                model.Add(cab_var == c_idx).OnlyEnforceIf(p)
                model.Add(cab_var != c_idx).OnlyEnforceIf(p.Not())
                c_int = model.NewOptionalIntervalVar(start_glob, length, end_glob, p, f"ci_{sid}_{idx}_{cname}")
                cabinet_intervals[cname].append((c_int, day_var, length, p))

            for sname in enrolled:
                s_int = model.NewIntervalVar(start_glob, length, end_glob, f"si_{sid}_{idx}_{sname}")
                student_intervals[sname].append((s_int, day_var, length))

            diff = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"diff_{sid}_{idx}")
            model.AddAbsEquality(diff, slot_var - subj.get("optimalSlot", 0))
            stud_pen = sum(student_importance[s] * student_size[s] for s in enrolled)
            base_penalties.append(diff * penalty_val * stud_pen * stud_weight)

            class_vars[(sid, idx)] = {
                "day": day_var,
                "slot": slot_var,
                "teacher": teach_var,
                "cab": cab_var,
                "len": length,
                "students": enrolled,
                "size": class_size,
                "start": start_glob,
            }

    # at most one class of same subject per day and chronological order
    for sid, subj in subjects.items():
        cls_cnt = len(subj["classes"])
        for d_idx in range(len(days)):
            flags = []
            for idx in range(cls_cnt):
                b = model.NewBoolVar(f"{sid}_{idx}_d{d_idx}")
                model.Add(class_vars[(sid, idx)]["day"] == d_idx).OnlyEnforceIf(b)
                model.Add(class_vars[(sid, idx)]["day"] != d_idx).OnlyEnforceIf(b.Not())
                flags.append(b)
            model.Add(sum(flags) <= 1)
        for i in range(1, cls_cnt):
            model.Add(class_vars[(sid, i)]["day"] > class_vars[(sid, i - 1)]["day"])

    # resource conflicts via NoOverlap
    for tname, items in teacher_intervals.items():
        model.AddNoOverlap([it[0] for it in items])
    for cname, items in cabinet_intervals.items():
        model.AddNoOverlap([it[0] for it in items])
    for sname, items in student_intervals.items():
        model.AddNoOverlap([it[0] for it in items])

    # compute gap variables using working span
    teacher_gap_vars = []
    for tname, items in teacher_intervals.items():
        for d_idx, day in enumerate(days):
            start_var = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"t_{tname}_{d_idx}_start")
            end_var = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"t_{tname}_{d_idx}_end")
            pres = []
            lengths = []
            for it, dv, l, p in items:
                b = model.NewBoolVar(f"use_{tname}_{d_idx}_{id(it)}")
                model.Add(dv == d_idx).OnlyEnforceIf(b)
                model.Add(dv != d_idx).OnlyEnforceIf(b.Not())
                model.Add(p == 1).OnlyEnforceIf(b)
                model.Add(p == 0).OnlyEnforceIf(b.Not())
                pres.append(b)
                lengths.append(l * b)
                model.Add(start_var <= it.StartExpr()).OnlyEnforceIf(b)
                model.Add(end_var >= it.EndExpr()).OnlyEnforceIf(b)
            if pres:
                total_len = sum(lengths)
                span = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"span_t_{tname}_{d_idx}")
                model.Add(span == end_var - start_var)
                gap = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"gap_t_{tname}_{d_idx}")
                model.Add(gap == span - total_len)
                teacher_gap_vars.append((gap, tname))

    student_gap_vars = []
    for sname, items in student_intervals.items():
        for d_idx, day in enumerate(days):
            start_var = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"s_{sname}_{d_idx}_start")
            end_var = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"s_{sname}_{d_idx}_end")
            pres = []
            lengths = []
            for it, dv, l in items:
                b = model.NewBoolVar(f"use_{sname}_{d_idx}_{id(it)}")
                model.Add(dv == d_idx).OnlyEnforceIf(b)
                model.Add(dv != d_idx).OnlyEnforceIf(b.Not())
                pres.append(b)
                lengths.append(l * b)
                model.Add(start_var <= it.StartExpr()).OnlyEnforceIf(b)
                model.Add(end_var >= it.EndExpr()).OnlyEnforceIf(b)
            if pres:
                total_len = sum(lengths)
                span = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"span_s_{sname}_{d_idx}")
                model.Add(span == end_var - start_var)
                gap = model.NewIntVar(0, max(len(d["slots"]) for d in days), f"gap_s_{sname}_{d_idx}")
                model.Add(gap == span - total_len)
                student_gap_vars.append((gap, sname))

    base_obj = sum(base_penalties)
    gap_teacher_expr = sum(gap_teacher_val * teacher_importance[t] * teach_weight * g for g, t in teacher_gap_vars)
    gap_student_expr = sum(gap_student_val * student_importance[s] * student_size[s] * stud_weight * g for g, s in student_gap_vars)
    model.Minimize(base_obj + gap_teacher_expr + gap_student_expr)

    solver = cp_model.CpSolver()
    params = cfg.get("model", {})
    solver.parameters.max_time_in_seconds = params.get("maxTime", 300)
    solver.parameters.num_search_workers = params.get("workers", 10)
    solver.parameters.log_search_progress = params.get("showProgress", False)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for (sid, idx), vars in class_vars.items():
        day_idx = solver.Value(vars["day"])
        start_slot = solver.Value(vars["slot"])
        teacher_name = teacher_names[solver.Value(vars["teacher"])]
        cab_name = list(cabinets)[solver.Value(vars["cab"])]
        day_name = days[day_idx]["name"]
        for s in range(start_slot, start_slot + vars["len"]):
            schedule[day_name][s].append({
                "subject": sid,
                "teacher": teacher_name,
                "cabinet": cab_name,
                "students": vars["students"],
                "size": vars["size"],
                "start": start_slot,
                "length": vars["len"],
            })

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
