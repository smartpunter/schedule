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
    penalty_val = cfg.get("penalties", {}).get("unoptimalSlot", [0])[0]
    settings = cfg.get("settings", {})
    stud_weight = settings.get("studentsPenaltyWeight", [1])[0]
    default_student_imp = settings.get("defaultStudentImportance", [0])[0]
    student_importance = {
        s["name"]: s.get("importance", default_student_imp) for s in students
    }

    # candidate variables for each subject class
    candidates: Dict[tuple, List[Dict[str, Any]]] = {}

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
            for day in days:
                dname = day["name"]
                slots = day["slots"]
                for start in slots:
                    if start + length - 1 > slots[-1]:
                        continue
                    for teacher in allowed_teachers:
                        for cab in allowed_cabinets:
                            if cabinets[cab]["capacity"] < class_size:
                                continue
                            var = model.NewBoolVar(
                                f"x_{sid}_{idx}_{dname}_{start}_{teacher}_{cab}"
                            )
                            diff = abs(start - subj.get("optimalSlot", 0))
                            stud_pen = sum(student_importance[s] for s in enrolled)
                            cand_list.append(
                                {
                                    "var": var,
                                    "day": dname,
                                    "start": start,
                                    "teacher": teacher,
                                    "cabinet": cab,
                                    "length": length,
                                    "size": class_size,
                                    "students": enrolled,
                                    "penalty": diff * penalty_val * stud_pen * stud_weight,
                                }
                            )
            if not cand_list:
                raise RuntimeError(f"No slot for subject {sid} class {idx}")
            model.Add(sum(c["var"] for c in cand_list) == 1)
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

    # teacher/student/cabinet conflicts
    for day in days:
        dname = day["name"]
        for slot in day["slots"]:
            for teacher in teacher_names:
                involved = []
                for (sid, idx), cand_list in candidates.items():
                    for c in cand_list:
                        if (
                            c["teacher"] == teacher
                            and c["day"] == dname
                            and slot in range(c["start"], c["start"] + c["length"])
                        ):
                            involved.append(c["var"])
                if involved:
                    model.Add(sum(involved) <= 1)

            for cab in cabinets:
                involved = []
                for cand_list in candidates.values():
                    for c in cand_list:
                        if (
                            c["cabinet"] == cab
                            and c["day"] == dname
                            and slot in range(c["start"], c["start"] + c["length"])
                        ):
                            involved.append(c["var"])
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

    # objective
    model.Minimize(
        sum(c["penalty"] * c["var"] for cand_list in candidates.values() for c in cand_list)
    )

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
        for c in cand_list:
            if solver.Value(c["var"]):
                for s in range(c["start"], c["start"] + c["length"]):
                    schedule[c["day"]][s].append(
                        {
                            "subject": sid,
                            "teacher": c["teacher"],
                            "cabinet": c["cabinet"],
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
                    p = penalties_cfg.get("gapStudent", 0) * student_importance[sname] * students_w
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
                    p = base * student_importance[sname] * students_w
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
