import json
import sys
from typing import Dict, List, Any


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
    """Construct naive schedule ignoring penalties."""
    days = cfg["days"]
    subjects = cfg["subjects"]
    teachers = cfg.get("teachers", [])
    students = cfg.get("students", [])
    cabinets = cfg.get("cabinets", {})

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

    schedule = _init_schedule(days)

    teacher_busy = {
        t["name"]: {day["name"]: {s: False for s in day["slots"]} for day in days}
        for t in teachers
    }
    student_busy = {
        stu["name"]: {day["name"]: {s: False for s in day["slots"]} for day in days}
        for stu in students
    }
    cabinet_busy = {
        cid: {day["name"]: {s: False for s in day["slots"]} for day in days}
        for cid in cabinets
    }

    for sid, subj in subjects.items():
        allowed_teachers = subject_teachers.get(sid)
        if not allowed_teachers:
            raise ValueError(f"No teacher available for subject {sid}")
        allowed_cabinets = subj.get("cabinets", list(cabinets))
        class_lengths = subj["classes"]
        enrolled = students_by_subject.get(sid, [])
        class_size = sum(student_size[s] for s in enrolled)

        used_days = set()
        for idx, length in enumerate(class_lengths):
            placed = False
            for day in days:
                dname = day["name"]
                if dname in used_days:
                    continue
                slots = day["slots"]
                for start in slots:
                    if start + length - 1 > slots[-1]:
                        continue
                    span = range(start, start + length)
                    for teacher in allowed_teachers:
                        if any(teacher_busy[teacher][dname][s] for s in span):
                            continue
                        for cab in allowed_cabinets:
                            if cabinets[cab]["capacity"] < class_size:
                                continue
                            if any(cabinet_busy[cab][dname][s] for s in span):
                                continue
                            if any(
                                student_busy[stu][dname][s]
                                for stu in enrolled
                                for s in span
                            ):
                                continue
                            # assign class
                            info = {
                                "subject": sid,
                                "teacher": teacher,
                                "cabinet": cab,
                                "students": enrolled,
                            }
                            for s in span:
                                schedule[dname][s].append(info)
                                teacher_busy[teacher][dname][s] = True
                                cabinet_busy[cab][dname][s] = True
                                for stu in enrolled:
                                    student_busy[stu][dname][s] = True
                            used_days.add(dname)
                            placed = True
                            break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                raise RuntimeError(f"Cannot place subject {sid} class {idx}")

    return schedule


def solve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate schedule and wrap it in export format."""
    schedule = build_model(cfg)
    export = {"days": []}
    for day in cfg["days"]:
        name = day["name"]
        slots = [schedule[name][s] for s in day["slots"]]
        export["days"].append({"name": name, "slots": slots})
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
