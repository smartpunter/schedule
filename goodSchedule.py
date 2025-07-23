import json
import os
from typing import Any, Dict, List, Tuple

from ortools.sat.python import cp_model

from newSchedule import load_config, _init_schedule


def _slot_index(days: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Return mapping day->offset and index->day"""
    offsets = {}
    day_ofs = {}
    total = 0
    for idx, d in enumerate(days):
        offsets[d["name"]] = total
        day_ofs[total] = d["name"]
        total += len(d["slots"])
    return offsets, day_ofs, total

def _entity_slots(entity: Dict[str, Any], days: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Return allowed slots per day for teacher or student."""
    allow = entity.get("allowedSlots")
    forbid = entity.get("forbiddenSlots")
    result: Dict[str, List[int]] = {}
    for day in days:
        name = day["name"]
        slots = set(day["slots"])
        if allow is not None:
            if name in allow:
                al = allow[name]
                slots = set(slots if al is None else al)
            else:
                slots = set()
        if forbid is not None and name in forbid:
            fb = forbid[name]
            if not fb:
                slots = set()
            else:
                slots -= set(fb)
        result[name] = sorted(slots)
    return result

def build_schedule(cfg: Dict[str, Any]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    days = cfg["days"]
    subjects = cfg["subjects"]
    teachers = cfg.get("teachers", [])
    students = cfg.get("students", [])
    cabinets = cfg.get("cabinets", {})

    offsets, day_ofs, total_slots = _slot_index(days)

    teacher_map = {t["name"]: set(t["subjects"]) for t in teachers}
    student_subjects = {s["name"]: set(s["subjects"]) for s in students}
    student_size = {s["name"]: int(s.get("group", 1)) for s in students}

    teacher_slots = {t["name"]: _entity_slots(t, days) for t in teachers}
    student_slots = {s["name"]: _entity_slots(s, days) for s in students}

    subjects_by_student: Dict[str, List[str]] = {}
    for s in students:
        for sid in s["subjects"]:
            subjects_by_student.setdefault(sid, []).append(s["name"])

    model = cp_model.CpModel()

    class_vars = {}
    class_lengths = {}
    class_students = {}
    teacher_assign: Dict[Tuple[str, int], Dict[str, cp_model.BoolVar]] = {}
    cabinet_assign: Dict[Tuple[str, int], Dict[str, cp_model.BoolVar]] = {}
    teacher_intervals = {t["name"]: [] for t in teachers}
    cabinet_intervals = {c: [] for c in cabinets}
    student_intervals = {s["name"]: [] for s in students}

    for sid, subj in subjects.items():
        lengths = subj.get("classes", [])
        allowed_cabs = subj.get("cabinets", list(cabinets))
        allowed_teach = [t for t in teacher_map if sid in teacher_map[t]]
        if not allowed_teach:
            raise ValueError(f"No teacher for subject {sid}")
        students_enrolled = subjects_by_student.get(sid, [])
        class_size = sum(student_size[s] for s in students_enrolled)
        for idx, length in enumerate(lengths):
            key = (sid, idx)
            class_lengths[key] = length
            class_students[key] = students_enrolled
            candidates = []
            for day in days:
                dname = day["name"]
                slots = day["slots"]
                last = slots[-1]
                for st in slots:
                    if st + length - 1 > last:
                        continue
                    ok = True
                    for stu in students_enrolled:
                        for s in range(st, st + length):
                            if s not in student_slots[stu][dname]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    candidates.append((dname, st))
            if not candidates:
                raise RuntimeError(f"No slots for class {sid} {idx}")
            choice = [model.NewBoolVar(f"ch_{sid}_{idx}_{i}") for i in range(len(candidates))]
            model.AddExactlyOne(choice)
            start_var = model.NewIntVar(0, total_slots - 1, f"start_{sid}_{idx}")
            model.Add(sum(choice[i] * (offsets[candidates[i][0]] + candidates[i][1]) for i in range(len(candidates))) == start_var)
            class_vars[key] = (start_var, choice, candidates)

            req_t = int(subj.get("requiredTeachers", 1))
            teacher_vars = {}
            for t in allowed_teach:
                var = model.NewBoolVar(f"teach_{sid}_{idx}_{t}")
                teacher_vars[t] = var
                for i, cand in enumerate(candidates):
                    glob = offsets[cand[0]] + cand[1]
                    for s in range(glob, glob + length):
                        if s - offsets[cand[0]] not in teacher_slots[t][cand[0]]:
                            model.Add(var + choice[i] <= 1)
                interval = model.NewOptionalIntervalVar(start_var, length, start_var + length, var, f"tiv_{sid}_{idx}_{t}")
                teacher_intervals[t].append(interval)
            model.Add(sum(teacher_vars.values()) == req_t)
            teacher_assign[key] = teacher_vars

            req_c = int(subj.get("requiredCabinets", 1))
            cabin_vars = {}
            for c in allowed_cabs:
                var = model.NewBoolVar(f"cab_{sid}_{idx}_{c}")
                cabin_vars[c] = var
                interval = model.NewOptionalIntervalVar(start_var, length, start_var + length, var, f"civ_{sid}_{idx}_{c}")
                cabinet_intervals[c].append(interval)
            model.Add(sum(cabin_vars.values()) == req_c)
            cabinet_assign[key] = cabin_vars

            for stu in students_enrolled:
                interval = model.NewIntervalVar(start_var, length, start_var + length, f"siv_{sid}_{idx}_{stu}")
                student_intervals[stu].append(interval)

    # NoOverlap constraints
    for t, ivs in teacher_intervals.items():
        if ivs:
            model.AddNoOverlap(ivs)
    for c, ivs in cabinet_intervals.items():
        if ivs:
            model.AddNoOverlap(ivs)
    for s, ivs in student_intervals.items():
        if ivs:
            model.AddNoOverlap(ivs)

    # objective: minimise deviation from optimal slot
    penalties = cfg.get("penalties", {})
    unopt = penalties.get("unoptimalSlot", [0])[0]
    defaults = cfg.get("defaults", {})
    teacher_imp_def = defaults.get("teacherImportance", [1])[0]
    student_imp_def = defaults.get("studentImportance", [0])[0]
    teacher_imp = {t["name"]: t.get("importance", teacher_imp_def) for t in teachers}
    student_imp = {s["name"]: s.get("importance", student_imp_def) for s in students}
    teach_weight = cfg.get("settings", {}).get("teacherAsStudents", [15])[0]

    obj_terms = []
    for (sid, idx), (start_var, choice, cand) in class_vars.items():
        opt = subjects[sid].get("optimalSlot", defaults.get("optimalSlot", [0])[0])
        size = sum(student_size[s] for s in class_students[(sid, idx)])
        for i, (dname, slot) in enumerate(cand):
            diff = abs(slot - opt)
            weight = unopt * diff * size
            obj_terms.append(choice[i] * weight)
    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    params = cfg.get("model", {})
    solver.parameters.max_time_in_seconds = params.get("maxTime", [10800])[0]
    workers = params.get("workers", [None])[0]
    if workers:
        solver.parameters.num_search_workers = workers
    solver.parameters.log_search_progress = params.get("showProgress", [True])[0]

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for (sid, idx), (start_var, choice, cand) in class_vars.items():
        sel = None
        for i in range(len(cand)):
            if solver.Value(choice[i]):
                sel = cand[i]
                break
        if sel is None:
            continue
        day = sel[0]
        slot = sel[1]
        length = class_lengths[(sid, idx)]
        teachers_used = [t for t, var in teacher_assign[(sid, idx)].items() if solver.Value(var)]
        cabinets_used = [c for c, var in cabinet_assign[(sid, idx)].items() if solver.Value(var)]
        entry = {
            "subject": sid,
            "teachers": teachers_used,
            "cabinets": cabinets_used,
            "students": class_students[(sid, idx)],
            "size": sum(student_size[s] for s in class_students[(sid, idx)]),
            "start": slot,
            "length": length,
        }
        for s in range(slot, slot + length):
            schedule[day][s].append(entry)
    return schedule


def main() -> None:
    cfg_path = "schedule-config.json"
    out_path = "schedule.json"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)
    schedule = build_schedule(cfg)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(schedule, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
