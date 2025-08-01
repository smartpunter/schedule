import copy
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Set

try:
    import json5  # type: ignore

    _json_loader = json5.load
except Exception:  # pragma: no cover - fallback when json5 is missing
    _json_loader = json.load
from datetime import datetime

from ortools.sat.python import cp_model

DEFAULT_MAX_TIME = 10800  # 3 hours
DEFAULT_SHOW_PROGRESS = True
DEFAULT_WORKERS = max(os.cpu_count() - 2, 4) if os.cpu_count() else 4


def _detect_duplicates(
    entities: List[Dict[str, Any]], key_fields: List[str]
) -> List[List[str]]:
    """Return lists of entity names that share identical parameters."""
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for ent in entities:
        key = tuple((f, json.dumps(ent.get(f), sort_keys=True)) for f in key_fields)
        groups[key].append(ent.get("name", ""))
    return [names for names in groups.values() if len(names) > 1]


def load_config(path: str = "schedule-config.json") -> Dict[str, Any]:
    """Load configuration file and apply defaults.

    This function supports JSON files with comments if the ``json5`` module
    is installed. Otherwise it falls back to the standard ``json`` loader.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = _json_loader(fh)

    settings = data.get("settings", {})
    defaults = data.get("defaults", {})
    default_teacher_imp = defaults.get("teacherImportance", [1])[0]
    default_student_imp = defaults.get("studentImportance", [0])[0]
    default_opt_slot = defaults.get("optimalSlot", [0])[0]
    default_permutations = defaults.get("permutations", [True])[0]
    default_avoid_consecutive = defaults.get("avoidConsecutive", [True])[0]
    default_teacher_arr = defaults.get("teacherArriveEarly", [False])[0]
    default_student_arr = defaults.get("studentArriveEarly", [True])[0]

    teachers_raw = data.get("teachers", {})
    teachers_list = []
    if isinstance(teachers_raw, dict):
        for name, info in teachers_raw.items():
            entry = {"name": name, **(info or {})}
            entry.setdefault("importance", default_teacher_imp)
            entry.setdefault("arriveEarly", default_teacher_arr)
            entry.setdefault("subjects", [])
            teachers_list.append(entry)
    else:
        for entry in teachers_raw:
            entry.setdefault("importance", default_teacher_imp)
            entry.setdefault("arriveEarly", default_teacher_arr)
            entry.setdefault("subjects", [])
            teachers_list.append(entry)
    data["teachers"] = teachers_list

    students_list = []
    for student in data.get("students", []):
        student.setdefault("importance", default_student_imp)
        student.setdefault("group", 1)
        student.setdefault("arriveEarly", default_student_arr)
        student.setdefault("optionalSubjects", [])
        students_list.append(student)
    data["students"] = students_list

    teacher_lookup = {t["name"]: t for t in teachers_list}
    for sid, subj in data.get("subjects", {}).items():
        subj.setdefault("optimalSlot", default_opt_slot)
        subj.setdefault("allowPermutations", default_permutations)
        subj.setdefault("avoidConsecutive", default_avoid_consecutive)
        if len(set(subj.get("classes", []))) <= 1:
            subj["allowPermutations"] = False
        if "cabinets" not in subj:
            subj["cabinets"] = list(data.get("cabinets", {}))
        subj.setdefault("primaryTeachers", [])
        subj.setdefault("requiredTeachers", 1)
        subj.setdefault("requiredCabinets", 1)
        for tname in subj.get("teachers", []):
            if tname in teacher_lookup:
                teacher_lookup[tname].setdefault("subjects", []).append(sid)

    lessons_parsed = []
    for item in data.get("lessons", []):
        if not isinstance(item, list) or len(item) < 4:
            raise ValueError(
                "Lesson entry must contain at least day, slot, subject and cabinet(s)"
            )

        day, slot, subject_id, cabinet = item[:4]
        teachers = None
        length = None
        if len(item) == 5:
            if isinstance(item[4], int):
                length = item[4]
            else:
                teachers = item[4]
        elif len(item) >= 6:
            teachers = item[4]
            length = item[5]
        if len(item) > 6:
            raise ValueError("Lesson entry has too many elements")

        if isinstance(cabinet, str):
            cabinet = [cabinet]
        lessons_parsed.append(
            {
                "day": day,
                "slot": int(slot),
                "subject": subject_id,
                "length": int(length) if length is not None else None,
                "cabinets": cabinet,
                "teachers": list(teachers) if teachers is not None else None,
            }
        )
    data["lessons"] = lessons_parsed

    model_conf = data.get("model", {})
    model_conf.setdefault("maxTime", DEFAULT_MAX_TIME)
    model_conf.setdefault("workers", DEFAULT_WORKERS)
    model_conf.setdefault("showProgress", DEFAULT_SHOW_PROGRESS)
    data["model"] = model_conf

    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    """Ensure all referenced entities exist and slots are valid."""
    days = {d["name"]: set(d["slots"]) for d in cfg.get("days", [])}
    subjects = cfg.get("subjects", {})
    teachers = {t["name"] for t in cfg.get("teachers", [])}
    cabinets = set(cfg.get("cabinets", {}))

    for stu in cfg.get("students", []):
        for sid in stu.get("subjects", []):
            if sid not in subjects:
                raise ValueError(
                    f"Student {stu.get('name')} references unknown subject '{sid}'"
                )
        for sid in stu.get("optionalSubjects", []):
            if sid not in subjects:
                raise ValueError(
                    f"Student {stu.get('name')} references unknown optional subject '{sid}'"
                )

    for sid, subj in subjects.items():
        for t in subj.get("teachers", []):
            if t not in teachers:
                raise ValueError(f"Subject {sid} references unknown teacher '{t}'")
        for cab in subj.get("cabinets", []):
            if cab not in cabinets:
                raise ValueError(f"Subject {sid} references unknown cabinet '{cab}'")
        for pt in subj.get("primaryTeachers", []):
            if pt not in teachers:
                raise ValueError(
                    f"Subject {sid} references unknown primary teacher '{pt}'"
                )

    for t in cfg.get("teachers", []):
        for sid in t.get("subjects", []):
            if sid not in subjects:
                raise ValueError(
                    f"Teacher {t['name']} assigned to unknown subject '{sid}'"
                )
        _check_slot_limits(days, t, f"Teacher {t['name']}")

    for s in cfg.get("students", []):
        _check_slot_limits(days, s, f"Student {s['name']}")

    for cname, cab in cfg.get("cabinets", {}).items():
        for sid in cab.get("allowedSubjects", []):
            if sid not in subjects:
                raise ValueError(f"Cabinet {cname} allows unknown subject '{sid}'")

    for lesson in cfg.get("lessons", []):
        # lessons may come from the raw config (list form) or from load_config
        if isinstance(lesson, dict):
            day = lesson.get("day")
            slot = lesson.get("slot")
            sid = lesson.get("subject")
            cabs = lesson.get("cabinets", [])
            teachers_part = lesson.get("teachers")
        else:
            if not isinstance(lesson, list) or len(lesson) < 4:
                raise ValueError(
                    "Lesson entry must contain at least day, slot, subject and cabinet(s)"
                )
            day, slot, sid, cabs = lesson[:4]
            teachers_part = None
            if len(lesson) == 5 and not isinstance(lesson[4], int):
                teachers_part = lesson[4]
            elif len(lesson) >= 6:
                teachers_part = lesson[4]

        if day not in days:
            raise ValueError(f"Unknown day '{day}' in lesson {lesson}")
        if int(slot) not in days[day]:
            raise ValueError(f"Slot {slot} not available on {day}")
        if sid not in subjects:
            raise ValueError(f"Unknown subject '{sid}' in lesson {lesson}")
        cab_list = [cabs] if isinstance(cabs, str) else list(cabs)
        for cab in cab_list:
            if cab not in cabinets:
                raise ValueError(f"Unknown cabinet '{cab}' in lesson {lesson}")
        if teachers_part is not None:
            t_list = (
                [teachers_part] if isinstance(teachers_part, str) else list(teachers_part)
            )
            for t in t_list:
                if t not in teachers:
                    raise ValueError(f"Unknown teacher '{t}' in lesson {lesson}")


def _check_slot_limits(
    days: Dict[str, Set[int]], entry: Dict[str, Any], name: str
) -> None:
    """Validate slot restrictions for a teacher or student."""
    for key in ("allowedSlots", "forbiddenSlots"):
        limits = entry.get(key, {})
        for day, slots in limits.items():
            if day not in days:
                raise ValueError(f"{name} uses unknown day '{day}' in {key}")
            if slots:
                for slot in slots:
                    if slot not in days[day]:
                        raise ValueError(
                            f"{name} uses invalid slot {slot} on {day} in {key}"
                        )


def _init_schedule(
    days: List[Dict[str, Any]],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Prepare empty schedule structure."""
    schedule = {}
    for day in days:
        name = day["name"]
        schedule[name] = {slot: [] for slot in day["slots"]}
    return schedule


def _calc_student_limits(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Set[int]]]:
    """Return allowed slots for each student."""
    limits: Dict[str, Dict[str, Set[int]]] = {}
    days = cfg.get("days", [])
    for stu in cfg.get("students", []):
        name = stu["name"]
        allow = stu.get("allowedSlots")
        forbid = stu.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        limits[name] = avail
    return limits



def _collect_optional_stats(
    schedule: Dict[str, Dict[int, List[Dict[str, Any]]]],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return attendance statistics for optional subjects without modifying the schedule."""
    students = cfg.get("students", [])
    student_names = [s["name"] for s in students]

    stats = {n: {"total": 0, "attended": 0, "subjects": {}} for n in student_names}

    for day in cfg.get("days", []):
        dname = day["name"]
        for slot in day["slots"]:
            for cls in schedule[dname][slot]:
                if slot != cls.get("start"):
                    continue
                sid = cls["subject"]
                length = cls.get("length", 1)
                for stu in students:
                    name = stu["name"]
                    if sid in stu.get("optionalSubjects", []):
                        subj = stats[name]["subjects"].setdefault(
                            sid, {"total": 0, "attended": 0}
                        )
                        subj["total"] += length
                        stats[name]["total"] += length
                        attended = len(cls.get("slotStudents", {}).get(name, set()))
                        subj["attended"] += attended
                        stats[name]["attended"] += attended

    return stats


def _deduplicate_schedule(schedule: Dict[str, Dict[int, List[Dict[str, Any]]]]) -> None:
    """Remove duplicate class entries from each schedule slot."""
    for day_slots in schedule.values():
        for slot, classes in day_slots.items():
            unique = []
            seen = set()
            for cls in classes:
                key = (
                    cls.get("subject"),
                    tuple(sorted(cls.get("teachers", []))),
                    tuple(sorted(cls.get("cabinets", []))),
                    cls.get("start"),
                    cls.get("length"),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique.append(cls)
            day_slots[slot] = unique




def build_fast_model(cfg: Dict[str, Any]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Build simplified schedule focusing on last lesson time."""
    days = cfg["days"]
    subjects = cfg["subjects"]
    teachers = cfg.get("teachers", [])
    students = cfg.get("students", [])
    cabinets = cfg.get("cabinets", {})

    teacher_names = [t["name"] for t in teachers]
    teacher_map = {t["name"]: set(t.get("subjects", [])) for t in teachers}

    teacher_limits: Dict[str, Dict[str, Set[int]]] = {}
    for t in teachers:
        name = t["name"]
        allow = t.get("allowedSlots")
        forbid = t.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        teacher_limits[name] = avail

    student_limits: Dict[str, Dict[str, Set[int]]] = {}
    students_by_subject: Dict[str, List[str]] = {}
    optional_by_subject: Dict[str, List[str]] = {}
    student_size: Dict[str, int] = {}
    for stu in students:
        name = stu["name"]
        allow = stu.get("allowedSlots")
        forbid = stu.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        student_limits[name] = avail
        student_size[name] = int(stu.get("group", 1))
        for sid in stu.get("subjects", []):
            students_by_subject.setdefault(sid, []).append(name)
        for sid in stu.get("optionalSubjects", []):
            optional_by_subject.setdefault(sid, []).append(name)

    # mapping from subject class to fixed lesson info
    day_lookup = {d["name"]: idx for idx, d in enumerate(days)}
    fixed_map: Dict[tuple, Dict[str, Any]] = {}
    remaining: Dict[str, List[int]] = {
        sid: list(range(len(subj.get("classes", [])))) for sid, subj in subjects.items()
    }
    length_map: Dict[str, Dict[int, List[int]]] = {}
    for sid, subj in subjects.items():
        lm: Dict[int, List[int]] = defaultdict(list)
        for idx, ln in enumerate(subj.get("classes", [])):
            lm.setdefault(ln, []).append(idx)
        length_map[sid] = lm

    for entry in cfg.get("lessons", []):
        day = entry["day"]
        slot = int(entry["slot"])
        sid = entry["subject"]
        length = entry.get("length")
        if length is None:
            idx = None
            for i in list(remaining[sid]):
                if slot + subjects[sid]["classes"][i] - 1 <= max(days[day_lookup[day]]["slots"]):
                    idx = i
                    length = subjects[sid]["classes"][i]
                    break
            if idx is None:
                idx = remaining[sid][0]
                length = subjects[sid]["classes"][idx]
        else:
            length = int(length)
            cand = length_map[sid].get(length, [])
            idx = next((i for i in cand if i in remaining[sid]), None)
            if idx is None:
                raise ValueError(f"No class of length {length} left for subject {sid}")
        remaining[sid].remove(idx)
        cabinets_fixed = entry["cabinets"]
        if isinstance(cabinets_fixed, str):
            cabinets_fixed = [cabinets_fixed]
        teachers_fixed = entry.get("teachers")
        if isinstance(teachers_fixed, str):
            teachers_fixed = [teachers_fixed]
        fixed_map[(sid, idx)] = {
            "day": day,
            "day_idx": day_lookup[day],
            "start": slot,
            "length": length,
            "cabinets": cabinets_fixed,
            "teachers": teachers_fixed,
        }

    idx_to_day: List[int] = []
    idx_to_slot: List[int] = []
    idx_map: Dict[tuple, int] = {}
    for d_idx, day in enumerate(days):
        for sl in day["slots"]:
            idx_map[(d_idx, sl)] = len(idx_to_day)
            idx_to_day.append(d_idx)
            idx_to_slot.append(sl)

    model = cp_model.CpModel()

    class_info: Dict[tuple, Dict[str, Any]] = {}
    teacher_intervals: Dict[str, List[cp_model.IntervalVar]] = defaultdict(list)
    cabinet_intervals: Dict[str, List[cp_model.IntervalVar]] = defaultdict(list)
    student_intervals: Dict[str, List[cp_model.IntervalVar]] = defaultdict(list)

    for sid, subj in subjects.items():
        class_lengths = subj.get("classes", [])
        allowed_teachers = subj.get("teachers", [])
        required_teachers = int(subj.get("requiredTeachers", 1))
        allowed_cabs = subj.get("cabinets", list(cabinets))
        required_cabs = int(subj.get("requiredCabinets", 1))
        # make a copy to prevent aliasing with global student lists
        class_students = list(students_by_subject.get(sid, []))
        size = sum(student_size[s] for s in class_students)
        for idx, length in enumerate(class_lengths):
            key = (sid, idx)
            fixed = fixed_map.get(key)
            if fixed is not None:
                start_idx = idx_map[(fixed["day_idx"], fixed["start"])]
                start_var = model.NewConstant(start_idx)
                day_var = model.NewConstant(fixed["day_idx"])
                slot_var = model.NewConstant(fixed["start"])
                teachers_list = fixed.get("teachers")
                cabinets_list = fixed.get("cabinets")
            else:
                choices = []
                for d_idx, day in enumerate(days):
                    slots = day["slots"]
                    dname = day["name"]
                    for sl in slots:
                        if sl + length - 1 > slots[-1]:
                            continue
                        if not all(
                            sl + off in student_limits[s][dname]
                            for s in class_students
                            for off in range(length)
                        ):
                            continue
                        avail_teachers = [
                            t
                            for t in allowed_teachers
                            if all(
                                sl + off in teacher_limits[t][dname]
                                for off in range(length)
                            )
                        ]
                        if len(avail_teachers) < required_teachers:
                            continue
                        choices.append(idx_map[(d_idx, sl)])
                if not choices:
                    raise ValueError(f"No slots available for {sid} class {idx}")
                choice_var = model.NewIntVar(0, len(choices) - 1, f"choice_{sid}_{idx}")
                start_var = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(choices), f"start_{sid}_{idx}"
                )
                day_vals = [idx_to_day[c] for c in choices]
                slot_vals = [idx_to_slot[c] for c in choices]
                day_var = model.NewIntVar(0, len(days) - 1, f"day_{sid}_{idx}")
                slot_var = model.NewIntVar(0, max(idx_to_slot), f"slot_{sid}_{idx}")
                model.AddElement(choice_var, choices, start_var)
                model.AddElement(choice_var, day_vals, day_var)
                model.AddElement(choice_var, slot_vals, slot_var)
                teachers_list = None
                cabinets_list = None

            end_var = model.NewIntVar(0, len(idx_to_day) + max(class_lengths), f"end_{sid}_{idx}")
            model.Add(end_var == start_var + length)
            interval = model.NewIntervalVar(start_var, length, end_var, f"cls_{sid}_{idx}")

            teacher_vars = {}
            if teachers_list is None:
                teach_choices = [t for t in allowed_teachers if t in teacher_map]
                if len(teach_choices) < required_teachers:
                    raise ValueError(
                        f"Subject {sid} requires {required_teachers} teachers but only {len(teach_choices)} available"
                    )
                vars_list = []
                for t in teach_choices:
                    v = model.NewBoolVar(f"teach_{sid}_{idx}_{t}")
                    teacher_vars[t] = v
                    allowed = [c for c in range(len(idx_to_day)) if all(idx_to_slot[c] + off in teacher_limits[t][days[idx_to_day[c]]["name"]] for off in range(length))]
                    for c in range(len(idx_to_day)):
                        if c not in allowed:
                            model.Add(start_var != c).OnlyEnforceIf(v)
                    t_interval = model.NewOptionalIntervalVar(start_var, length, end_var, v, f"t_{sid}_{idx}_{t}")
                    teacher_intervals[t].append(t_interval)
                    vars_list.append(v)
                if not vars_list:
                    raise ValueError(f"No teacher for {sid}")
                model.Add(sum(vars_list) == required_teachers)
            else:
                for t in teachers_list:
                    v = model.NewConstant(1)
                    teacher_vars[t] = v
                    t_interval = model.NewOptionalIntervalVar(start_var, length, end_var, v, f"t_{sid}_{idx}_{t}")
                    teacher_intervals[t].append(t_interval)

            cabinet_vars = {}
            if cabinets_list is None:
                cab_choices = [
                    c
                    for c in allowed_cabs
                    if c in cabinets
                    and (
                        not cabinets[c].get("allowedSubjects")
                        or sid in cabinets[c]["allowedSubjects"]
                    )
                ]
                if len(cab_choices) < required_cabs:
                    raise ValueError(
                        f"Subject {sid} requires {required_cabs} cabinets but only {len(cab_choices)} available"
                    )
                caps = sorted((cabinets[c]["capacity"] for c in cab_choices), reverse=True)
                if sum(caps[:required_cabs]) < size:
                    raise ValueError(
                        f"Available cabinets for {sid} cannot fit class size {size}"
                    )
                vars_c = []
                for c in cab_choices:
                    v = model.NewBoolVar(f"cab_{sid}_{idx}_{c}")
                    cabinet_vars[c] = v
                    c_interval = model.NewOptionalIntervalVar(start_var, length, end_var, v, f"c_{sid}_{idx}_{c}")
                    cabinet_intervals[c].append(c_interval)
                    vars_c.append(v)
                if not vars_c:
                    raise ValueError(f"No cabinet for {sid}")
                model.Add(sum(vars_c) == required_cabs)
            else:
                for c in cabinets_list:
                    v = model.NewConstant(1)
                    cabinet_vars[c] = v
                    c_interval = model.NewOptionalIntervalVar(start_var, length, end_var, v, f"c_{sid}_{idx}_{c}")
                    cabinet_intervals[c].append(c_interval)

            for s in class_students:
                student_intervals[s].append(interval)

            class_info[key] = {
                "start": start_var,
                "day": day_var,
                "slot": slot_var,
                "length": length,
                "end_val": end_var,
                "teachers": teacher_vars,
                "cabinets": cabinet_vars,
                # store copy of students to prevent modifications from affecting others
                "students": list(class_students),
                "size": size,
            }

    for t in teacher_intervals:
        model.AddNoOverlap(teacher_intervals[t])
    for c in cabinet_intervals:
        model.AddNoOverlap(cabinet_intervals[c])
    for s in student_intervals:
        model.AddNoOverlap(student_intervals[s])

    max_slot = max(sl for d in days for sl in d["slots"]) + max(len(s.get("classes", [])) for s in subjects.values())
    penalties = []
    for t in teacher_names:
        for d_idx, day in enumerate(days):
            ends = []
            for key, info in class_info.items():
                if t in info["teachers"]:
                    tv = info["teachers"][t]
                    is_day = model.NewBoolVar(f"day_t_{key[0]}_{key[1]}_{t}_{d_idx}")
                    model.Add(info["day"] == d_idx).OnlyEnforceIf(is_day)
                    model.Add(info["day"] != d_idx).OnlyEnforceIf(is_day.Not())
                    active = model.NewBoolVar(f"act_t_{key[0]}_{key[1]}_{t}_{d_idx}")
                    model.AddMultiplicationEquality(active, [tv, is_day])
                    val = model.NewIntVar(0, max_slot + 1, f"end_t_{key[0]}_{key[1]}_{t}_{d_idx}")
                    model.Add(val == info["slot"] + info["length"]).OnlyEnforceIf(active)
                    model.Add(val == 0).OnlyEnforceIf(active.Not())
                    ends.append(val)
            if ends:
                last = model.NewIntVar(0, max_slot + 1, f"last_t_{t}_{d_idx}")
                model.AddMaxEquality(last, ends)
            else:
                last = model.NewConstant(0)
            penalties.append(last)
    for s in student_intervals:
        weight = student_size.get(s, 1)
        for d_idx, day in enumerate(days):
            ends = []
            for key, info in class_info.items():
                if s in info["students"]:
                    is_day = model.NewBoolVar(f"day_s_{key[0]}_{key[1]}_{s}_{d_idx}")
                    model.Add(info["day"] == d_idx).OnlyEnforceIf(is_day)
                    model.Add(info["day"] != d_idx).OnlyEnforceIf(is_day.Not())
                    val = model.NewIntVar(0, max_slot + 1, f"end_s_{key[0]}_{key[1]}_{s}_{d_idx}")
                    model.Add(val == info["slot"] + info["length"]).OnlyEnforceIf(is_day)
                    model.Add(val == 0).OnlyEnforceIf(is_day.Not())
                    ends.append(val)
            if ends:
                last = model.NewIntVar(0, max_slot + 1, f"last_s_{s}_{d_idx}")
                model.AddMaxEquality(last, ends)
            else:
                last = model.NewConstant(0)
            if weight > 1:
                scaled = model.NewIntVar(0, max_slot * weight + weight, f"sc_{s}_{d_idx}")
                model.AddMultiplicationEquality(scaled, [last, weight])
                penalties.append(scaled)
            else:
                penalties.append(last)

    model.Minimize(sum(penalties))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.get("model", {}).get("maxTime", [DEFAULT_MAX_TIME])[0]
    solver.parameters.num_search_workers = cfg.get("model", {}).get("workers", [DEFAULT_WORKERS])[0] or DEFAULT_WORKERS
    solver.parameters.log_search_progress = cfg.get("model", {}).get("showProgress", [DEFAULT_SHOW_PROGRESS])[0]

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for key, info in class_info.items():
        start = solver.Value(info["slot"])
        day_idx = solver.Value(info["day"])
        length = info["length"]
        dname = days[day_idx]["name"]
        teachers_assigned = [t for t, v in info["teachers"].items() if solver.Value(v)]
        cabinets_assigned = [c for c, v in info["cabinets"].items() if solver.Value(v)]
        cls_obj = {
            "subject": key[0],
            "teachers": teachers_assigned,
            "cabinets": cabinets_assigned,
            # copy to keep class attendance independent
            "students": list(info["students"]),
            "slotStudents": {
                st: set(range(start, start + length)) for st in info["students"]
            },
            "size": info["size"],
            "optionalSize": 0,
            "start": start,
            "length": length,
        }
        for s in range(start, start + length):
            schedule[dname][s].append(cls_obj)

    return schedule

def _prepare_fixed_classes(
    cfg: Dict[str, Any],
    teacher_limits: Dict[str, Dict[str, Set[int]]],
    teacher_map: Dict[str, Set[str]],
    student_limits: Dict[str, Dict[str, Set[int]]],
    students_by_subject: Dict[str, List[str]],
    student_size: Dict[str, int],
) -> Dict[tuple, Dict[str, Any]]:
    """Validate and convert fixed lessons configuration."""
    lessons = cfg.get("lessons", [])
    if not lessons:
        return {}

    day_lookup = {d["name"]: set(d["slots"]) for d in cfg["days"]}
    day_index = {d["name"]: idx for idx, d in enumerate(cfg["days"])}
    cabinets = cfg.get("cabinets", {})
    subjects = cfg["subjects"]

    # track available class indices by length for each subject
    length_map: Dict[str, Dict[int, List[int]]] = {}
    remaining: Dict[str, List[int]] = {}
    for sid, subj in subjects.items():
        lm: Dict[int, List[int]] = defaultdict(list)
        for idx, ln in enumerate(subj.get("classes", [])):
            lm[ln].append(idx)
        length_map[sid] = lm
        remaining[sid] = list(range(len(subj.get("classes", []))))

    fixed: Dict[tuple, Dict[str, Any]] = {}
    for entry in lessons:
        day = entry["day"]
        slot = int(entry["slot"])
        sid = entry["subject"]
        length_override = entry.get("length")
        rooms = entry["cabinets"]
        if isinstance(rooms, str):
            rooms = [rooms]
        tlist = entry.get("teachers")
        explicit_teachers = True
        if not tlist:
            explicit_teachers = False
            tlist = []
        elif isinstance(tlist, str):
            tlist = [tlist]

        if day not in day_lookup:
            raise ValueError(f"Unknown day '{day}' in lesson {entry}")
        if slot not in day_lookup[day]:
            raise ValueError(f"Slot {slot} not available on {day}")
        if sid not in subjects:
            raise ValueError(f"Unknown subject '{sid}' in lesson {entry}")
        subj = subjects[sid]

        if length_override is not None:
            length = int(length_override)
            choices = length_map[sid].get(length)
            while choices and choices[0] not in remaining[sid]:
                choices.pop(0)
            if not choices:
                raise ValueError(
                    f"No unused class of length {length} for subject {sid}"
                )
            idx = choices.pop(0)
            remaining[sid].remove(idx)
        else:
            if not remaining[sid]:
                raise ValueError(f"Too many fixed lessons for subject {sid}")
            last_slot = max(day_lookup[day])
            idx = None
            for cand in list(remaining[sid]):
                cand_len = subj["classes"][cand]
                if slot + cand_len - 1 <= last_slot:
                    idx = cand
                    break
            if idx is None:
                raise ValueError(
                    f"Lesson for {sid} starting at {day} slot {slot} exceeds day length"
                )
            remaining[sid].remove(idx)
            length = subj["classes"][idx]
            length_map[sid][length].remove(idx)
        last_slot = max(day_lookup[day])
        if slot + length - 1 > last_slot:
            raise ValueError(
                f"Lesson for {sid} starting at {day} slot {slot} exceeds day length"
            )

        required_cabs = int(subj.get("requiredCabinets", 1))
        if len(rooms) != required_cabs:
            raise ValueError(
                f"Subject {sid} requires {required_cabs} cabinets, got {len(rooms)}"
            )
        class_size = sum(student_size[s] for s in students_by_subject.get(sid, []))
        for room in rooms:
            if room not in cabinets:
                raise ValueError(f"Unknown cabinet '{room}' in lesson {entry}")
            if room not in subj.get("cabinets", list(cabinets)):
                raise ValueError(f"Cabinet '{room}' not allowed for subject {sid}")
            allowed = cabinets[room].get("allowedSubjects")
            if allowed and sid not in allowed:
                raise ValueError(f"Subject {sid} not permitted in cabinet '{room}'")
        total_capacity = sum(cabinets[r]["capacity"] for r in rooms)
        if total_capacity < class_size:
            raise ValueError(
                f"Cabinets {rooms} too small for subject {sid} (size {class_size})"
            )

        required = int(subj.get("requiredTeachers", 1))
        if explicit_teachers:
            if len(tlist) != required:
                raise ValueError(
                    f"Subject {sid} requires {required} teachers, got {len(tlist)}"
                )

            for t in tlist:
                if t not in teacher_map or sid not in teacher_map[t]:
                    raise ValueError(f"Teacher {t} cannot teach subject {sid}")
                for off in range(length):
                    if slot + off not in teacher_limits[t][day]:
                        raise ValueError(
                            f"Teacher {t} not available for entire duration starting at {day} slot {slot} for subject {sid}"
                        )
            available_teachers = tlist
        else:
            available_teachers = [
                t
                for t in teacher_map
                if sid in teacher_map[t]
                and all(s in teacher_limits[t][day] for s in range(slot, slot + length))
            ]
            if len(available_teachers) < required:
                raise ValueError(
                    f"Subject {sid} requires {required} teachers, only {len(available_teachers)} available"
                )

        for stu in students_by_subject.get(sid, []):
            for off in range(length):
                if slot + off not in student_limits[stu][day]:
                    raise ValueError(
                        f"Student {stu} not available for entire duration starting at {day} slot {slot} for subject {sid}"
                    )

        primary = set(subj.get("primaryTeachers", []))
        if explicit_teachers and primary and not primary.issubset(set(tlist)):
            missing = ", ".join(sorted(primary - set(tlist)))
            raise ValueError(
                f"Lesson for subject {sid} missing primary teacher(s): {missing}"
            )
        if (
            not explicit_teachers
            and primary
            and not primary.issubset(set(available_teachers))
        ):
            missing = ", ".join(sorted(primary - set(available_teachers)))
            raise ValueError(
                f"Primary teacher(s) {missing} not available for fixed lesson of {sid}"
            )

        fixed[(sid, idx)] = {
            "day": day,
            "day_idx": day_index[day],
            "start": slot,
            "length": length,
            "cabinets": rooms,
            "teachers": tlist if explicit_teachers else None,
            "available_teachers": available_teachers,
        }

    return fixed


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

    # calculate allowed slots for every teacher
    teacher_limits: Dict[str, Dict[str, Set[int]]] = {}
    for t in teachers:
        name = t["name"]
        allow = t.get("allowedSlots")
        forbid = t.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        teacher_limits[name] = avail

    # calculate allowed slots for every student
    student_limits: Dict[str, Dict[str, Set[int]]] = {}
    students_by_subject: Dict[str, List[str]] = {}
    optional_by_subject: Dict[str, List[str]] = {}
    student_size: Dict[str, int] = {}
    student_arrive = {s["name"]: bool(s.get("arriveEarly", True)) for s in students}
    for stu in students:
        name = stu["name"]
        allow = stu.get("allowedSlots")
        forbid = stu.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        student_limits[name] = avail

        student_size[name] = int(stu.get("group", 1))
        for sid in stu["subjects"]:
            students_by_subject.setdefault(sid, []).append(name)
        for sid in stu.get("optionalSubjects", []):
            optional_by_subject.setdefault(sid, []).append(name)

    fixed_classes = _prepare_fixed_classes(
        cfg,
        teacher_limits,
        teacher_map,
        student_limits,
        students_by_subject,
        student_size,
    )

    model = cp_model.CpModel()
    penalties = cfg.get("penalties", {})
    penalty_val = penalties.get("unoptimalSlot", [0])[0]
    gap_teacher_val = penalties.get("gapTeacher", [0])[0]
    gap_student_val = penalties.get("gapStudent", [0])[0]
    consecutive_pen_val = penalties.get("consecutiveClass", [0])[0]
    teacher_streak_list = penalties.get("teacherLessonStreak", [[]])[0]
    student_streak_list = penalties.get("studentLessonStreak", [[]])[0]
    settings = cfg.get("settings", {})
    teacher_as_students = settings.get("teacherAsStudents", [15])[0]
    duplicates_pen_val = settings.get("duplicatesPenalty", [0])[0]
    defaults = cfg.get("defaults", {})
    default_permutations = defaults.get("permutations", [True])[0]
    default_avoid_consecutive = defaults.get("avoidConsecutive", [True])[0]
    default_student_imp = defaults.get("studentImportance", [0])[0]
    default_teacher_imp = defaults.get("teacherImportance", [1])[0]
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
    avoid_map = {
        sid: subj.get("avoidConsecutive", default_avoid_consecutive)
        for sid, subj in subjects.items()
    }
    # helper mapping for allowed teacher and cabinet choices
    for sid, subj in subjects.items():
        allowed_teachers = subject_teachers.get(sid)
        if not allowed_teachers:
            raise ValueError(f"No teacher available for subject {sid}")
        # copy so later optional assignments do not mutate the shared list
        enrolled = list(students_by_subject.get(sid, []))
        optional_students = list(optional_by_subject.get(sid, []))
        class_size = sum(student_size[s] for s in enrolled)
        allowed_cabinets = [
            c
            for c in subj.get("cabinets", list(cabinets))
            if not cabinets.get(c, {}).get("allowedSubjects")
            or sid in cabinets[c]["allowedSubjects"]
        ]
        required_cabs = int(subj.get("requiredCabinets", 1))
        if len(allowed_cabinets) < required_cabs:
            raise ValueError(
                f"Subject {sid} needs {required_cabs} cabinets but only {len(allowed_cabinets)} provided"
            )
        caps = sorted((cabinets[c]["capacity"] for c in allowed_cabinets), reverse=True)
        if sum(caps[:required_cabs]) < class_size:
            raise ValueError(
                f"Available cabinets for {sid} cannot fit class size {class_size}"
            )
        class_lengths = subj["classes"]

        for idx, length in enumerate(class_lengths):
            key = (sid, idx)
            fixed = fixed_classes.get(key)
            cand_list = []
            if fixed is not None:
                var = model.NewBoolVar(f"x_{sid}_{idx}_{fixed['day']}_{fixed['start']}")
                model.Add(var == 1)
                # candidate presence is fixed for predefined lessons
                diff = abs(fixed["start"] - subj.get("optimalSlot", 0))
                teachers_for_pen = (
                    fixed["teachers"]
                    if fixed["teachers"] is not None
                    else fixed["available_teachers"]
                )
                stud_pen_map = {
                    s: diff * penalty_val * student_importance[s] * student_size[s]
                    for s in enrolled
                }
                teach_pen_map = {
                    t: diff * penalty_val * teacher_importance[t] * teacher_as_students
                    for t in teachers_for_pen
                }
                cand_list.append(
                    {
                        "var": var,
                        "day": fixed["day"],
                        "day_idx": fixed["day_idx"],
                        "start": fixed["start"],
                        "length": length,
                        "size": class_size,
                        # store a copy to avoid mutating the source list later
                        "students": list(enrolled),
                        "optional_students": list(optional_students),
                        "student_pen": stud_pen_map,
                        "teacher_pen": teach_pen_map,
                        "penalty": sum(stud_pen_map.values())
                        + sum(teach_pen_map.values()),
                        "available_teachers": fixed["available_teachers"],
                    }
                )
            else:
                for day_idx, day in enumerate(days):
                    dname = day["name"]
                    slots = day["slots"]
                    required = int(subj.get("requiredTeachers", 1))
                    for start in slots:
                        if start + length - 1 > slots[-1]:
                            continue
                        available = [
                            t
                            for t in allowed_teachers
                            if all(
                                s in teacher_limits[t][dname]
                                for s in range(start, start + length)
                            )
                        ]
                        if len(available) < required:
                            continue
                        if any(
                            not all(
                                s in student_limits[stu][dname]
                                for s in range(start, start + length)
                            )
                        for stu in enrolled
                        ):
                            continue
                        var = model.NewBoolVar(f"x_{sid}_{idx}_{dname}_{start}")
                        diff = abs(start - subj.get("optimalSlot", 0))
                        stud_pen_map = {
                            s: diff
                            * penalty_val
                            * student_importance[s]
                            * student_size[s]
                            for s in enrolled
                        }
                        teach_pen_map = {
                            t: diff
                            * penalty_val
                            * teacher_importance[t]
                            * teacher_as_students
                            for t in available
                        }
                        cand_list.append(
                            {
                                "var": var,
                                "day": dname,
                                "day_idx": day_idx,
                                "start": start,
                                "length": length,
                                "size": class_size,
                                # copy to decouple from shared enrollment list
                                "students": list(enrolled),
                                "optional_students": list(optional_students),
                                "student_pen": stud_pen_map,
                                "teacher_pen": teach_pen_map,
                                "penalty": sum(stud_pen_map.values())
                                + sum(teach_pen_map.values()),
                                "available_teachers": available,
                            }
                        )
            if not cand_list:
                raise RuntimeError(f"No slot for subject {sid} class {idx}")
            cand_list.sort(key=lambda c: c["penalty"])  # sort by penalty
            model.Add(sum(c["var"] for c in cand_list) == 1)
            day_var = model.NewIntVar(0, len(days) - 1, f"day_idx_{sid}_{idx}")
            model.Add(day_var == sum(c["day_idx"] * c["var"] for c in cand_list))
            class_day_idx[key] = day_var
            required = int(subj.get("requiredTeachers", 1))
            primary = set(subj.get("primaryTeachers", []))

            for cand in cand_list:
                if any(pt not in cand["available_teachers"] for pt in primary):
                    model.Add(cand["var"] == 0)

                teach_vars = {}
                for t in cand["available_teachers"]:
                    tv = model.NewBoolVar(
                        f"teach_{sid}_{idx}_{t}_{cand['day']}_{cand['start']}"
                    )
                    model.Add(tv <= cand["var"])
                    if fixed is not None and fixed.get("teachers") is not None:
                        if t in fixed["teachers"]:
                            model.Add(tv == cand["var"])
                        else:
                            model.Add(tv == 0)
                    elif t in primary:
                        model.Add(tv == cand["var"])
                    teach_vars[t] = tv
                if teach_vars:
                    model.Add(sum(teach_vars.values()) == required * cand["var"])
                cand["teacher_pres"] = teach_vars

                cab_vars = {}
                for c in allowed_cabinets:
                    cv = model.NewBoolVar(
                        f"cab_{sid}_{idx}_{c}_{cand['day']}_{cand['start']}"
                    )
                    model.Add(cv <= cand["var"])
                    if fixed is not None:
                        if c in fixed["cabinets"]:
                            model.Add(cv == cand["var"])
                        else:
                            model.Add(cv == 0)
                    cab_vars[c] = cv
                if cab_vars:
                    model.Add(sum(cab_vars.values()) == required_cabs * cand["var"])
                    model.Add(
                        sum(
                            cabinets[c]["capacity"] * cab_vars[c]
                            for c in allowed_cabinets
                        )
                        >= class_size * cand["var"]
                    )
                cand["cabinet_pres"] = cab_vars

                cand["interval"] = model.NewOptionalIntervalVar(
                    cand["start"],
                    cand["length"],
                    cand["start"] + cand["length"],
                    cand["var"],
                    f"int_{sid}_{idx}_{cand['day']}_{cand['start']}",
                )
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
                    c["var"] for c in candidates[(sid, idx)] if c["day"] == day["name"]
                )
            if vars_in_day:
                model.Add(sum(vars_in_day) <= 1)

        if not subj.get("allowPermutations", default_permutations):
            # enforce strict class order across days
            for idx in range(1, class_count):
                prev_k = (sid, idx - 1)
                curr_k = (sid, idx)
                model.Add(class_day_idx[curr_k] > class_day_idx[prev_k])

    # teacher, cabinet and student intervals with built-in constraints
    # store (interval, start, end, presence) tuples to keep bounds handy
    teacher_intervals = {t: defaultdict(list) for t in teacher_names}
    cabinet_intervals = {c: defaultdict(list) for c in cabinets}
    student_intervals = {s: defaultdict(list) for s in student_size}
    optional_exprs_map = {s: [] for s in student_size}

    for (sid, idx), cand_list in candidates.items():
        for cand in cand_list:
            base_int = cand["interval"]
            start = cand["start"]
            end = cand["start"] + cand["length"]
            for stu in cand["students"]:
                student_intervals[stu][cand["day"]].append(
                    (base_int, start, end, cand["var"])
                )
            for stu in cand.get("optional_students", []):
                for off in range(cand["length"]):
                    slot = start + off
                    attend = model.NewBoolVar(
                        f"opt_{sid}_{idx}_{stu}_{cand['day']}_{slot}"
                    )
                    model.Add(attend <= cand["var"])
                    if slot not in student_limits[stu][cand["day"]]:
                        model.Add(attend == 0)
                    interval = model.NewOptionalIntervalVar(
                        slot,
                        1,
                        slot + 1,
                        attend,
                        f"oint_{sid}_{idx}_{stu}_{cand['day']}_{slot}",
                    )
                    student_intervals[stu][cand["day"]].append((interval, slot, slot + 1, attend))
                    optional_exprs_map[stu].append(cand["var"] - attend)
                    cand.setdefault("optional_pres", {}).setdefault(stu, {})[slot] = attend
            for t, pres in cand.get("teacher_pres", {}).items():
                interval = model.NewOptionalIntervalVar(
                    start,
                    cand["length"],
                    end,
                    pres,
                    f"tint_{sid}_{idx}_{t}_{cand['day']}_{cand['start']}",
                )
                teacher_intervals[t][cand["day"]].append((interval, start, end, pres))
            for cab, cv in cand.get("cabinet_pres", {}).items():
                interval = model.NewOptionalIntervalVar(
                    cand["start"],
                    cand["length"],
                    cand["start"] + cand["length"],
                    cv,
                    f"cint_{sid}_{idx}_{cab}_{cand['day']}_{cand['start']}",
                )
                cabinet_intervals[cab][cand["day"]].append((interval, start, end, cv))

    for t, day_map in teacher_intervals.items():
        for ivs in day_map.values():
            if ivs and duplicates_pen_val <= 0:
                model.AddNoOverlap([iv[0] for iv in ivs])
    for c, day_map in cabinet_intervals.items():
        for ivs in day_map.values():
            if ivs:
                model.AddNoOverlap([iv[0] for iv in ivs])
    for s, day_map in student_intervals.items():
        for ivs in day_map.values():
            if ivs and duplicates_pen_val <= 0:
                model.AddNoOverlap([iv[0] for iv in ivs])

    # build slot variables for teachers and students based on intervals
    teacher_slot = {}
    student_slot = {}
    teacher_dup_exprs_map = {t: [] for t in teacher_names}
    student_dup_exprs_map = {s: [] for s in student_importance}
    for day in days:
        dname = day["name"]
        for slot in day["slots"]:
            for t in teacher_names:
                covering = [
                    info[3]
                    for info in teacher_intervals[t].get(dname, [])
                    if info[1] <= slot < info[2]
                ]
                if covering:
                    var = model.NewBoolVar(f"teach_{t}_{dname}_{slot}")
                    model.AddMaxEquality(var, covering)
                    if duplicates_pen_val > 0 and len(covering) > 1:
                        expr = (
                            sum(covering) - var
                        ) * duplicates_pen_val * teacher_importance[t]
                        teacher_dup_exprs_map[t].append(expr)
                else:
                    var = model.NewBoolVar(f"teach_{t}_{dname}_{slot}")
                    model.Add(var == 0)
                if slot not in teacher_limits[t][dname]:
                    model.Add(var == 0)
                teacher_slot[(t, dname, slot)] = var

            for stu in students:
                sname = stu["name"]
                covering = [
                    info[3]
                    for info in student_intervals.get(sname, {}).get(dname, [])
                    if info[1] <= slot < info[2]
                ]
                if covering:
                    var = model.NewBoolVar(f"stud_{sname}_{dname}_{slot}")
                    model.AddMaxEquality(var, covering)
                    if duplicates_pen_val > 0 and len(covering) > 1:
                        expr = (
                            sum(covering) - var
                        ) * duplicates_pen_val * student_importance[sname] * student_size[sname]
                        student_dup_exprs_map[sname].append(expr)
                else:
                    var = model.NewBoolVar(f"stud_{sname}_{dname}_{slot}")
                    model.Add(var == 0)
                if slot not in student_limits[sname][dname]:
                    model.Add(var == 0)
                student_slot[(sname, dname, slot)] = var

    # gap detection using simple before/after check
    teacher_gap_vars = []
    for t in teacher_names:
        arrive = teacher_arrive.get(t, False)
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            for idx, s in enumerate(slots):
                cur = teacher_slot[(t, dname, s)]
                before = model.NewBoolVar(f"before_t_{t}_{dname}_{s}")
                if idx > 0:
                    model.AddMaxEquality(
                        before,
                        [teacher_slot[(t, dname, k)] for k in slots[:idx]],
                    )
                else:
                    model.Add(before == (1 if arrive else 0))
                after = model.NewBoolVar(f"after_t_{t}_{dname}_{s}")
                if idx < len(slots) - 1:
                    model.AddMaxEquality(
                        after,
                        [teacher_slot[(t, dname, k)] for k in slots[idx + 1 :]],
                    )
                else:
                    model.Add(after == 0)
                g = model.NewBoolVar(f"gap_t_{t}_{dname}_{s}")
                model.Add(g <= before)
                model.Add(g <= after)
                model.Add(g + cur <= 1)
                model.Add(g >= before + after + (1 - cur) - 2)
                teacher_gap_vars.append((g, t))
    student_gap_vars = []
    for stu in students:
        sname = stu["name"]
        arrive = student_arrive.get(sname, True)
        for day in days:
            dname = day["name"]
            slots = day["slots"]
            for idx, s in enumerate(slots):
                cur = student_slot[(sname, dname, s)]
                before = model.NewBoolVar(f"before_s_{sname}_{dname}_{s}")
                if idx > 0:
                    model.AddMaxEquality(
                        before,
                        [student_slot[(sname, dname, k)] for k in slots[:idx]],
                    )
                else:
                    model.Add(before == (1 if arrive else 0))
                after = model.NewBoolVar(f"after_s_{sname}_{dname}_{s}")
                if idx < len(slots) - 1:
                    model.AddMaxEquality(
                        after,
                        [student_slot[(sname, dname, k)] for k in slots[idx + 1 :]],
                    )
                else:
                    model.Add(after == 0)
                g = model.NewBoolVar(f"gap_s_{sname}_{dname}_{s}")
                model.Add(g <= before)
                model.Add(g <= after)
                model.Add(g + cur <= 1)
                model.Add(g >= before + after + (1 - cur) - 2)
                student_gap_vars.append((g, sname))

    teacher_streak_exprs = {t: [] for t in teacher_names}
    if teacher_streak_list:
        t_inc = []
        prev = 0
        for val in teacher_streak_list:
            t_inc.append(max(0, int(val) - prev))
            prev = int(val)
        max_len = len(teacher_streak_list)
        for t in teacher_names:
            for day in days:
                dname = day["name"]
                slots = day["slots"]
                for idx, s in enumerate(slots):
                    for k in range(1, max_len + 1):
                        if idx < k - 1:
                            continue
                        parts = [
                            teacher_slot[(t, dname, slots[idx - j])] for j in range(k)
                        ]
                        st = model.NewBoolVar(f"tstreak_{t}_{dname}_{s}_{k}")
                        for p in parts:
                            model.Add(st <= p)
                        model.Add(st >= sum(parts) - k + 1)
                        if t_inc[k - 1] > 0:
                            teacher_streak_exprs[t].append(
                                st
                                * t_inc[k - 1]
                                * teacher_importance[t]
                                * teacher_as_students
                            )
    student_streak_exprs = {s: [] for s in student_importance}
    if student_streak_list:
        s_inc = []
        prev = 0
        for val in student_streak_list:
            s_inc.append(max(0, int(val) - prev))
            prev = int(val)
        max_len = len(student_streak_list)
        for sname in student_importance:
            for day in days:
                dname = day["name"]
                slots = day["slots"]
                for idx, s in enumerate(slots):
                    for k in range(1, max_len + 1):
                        if idx < k - 1:
                            continue
                        parts = [
                            student_slot[(sname, dname, slots[idx - j])]
                            for j in range(k)
                        ]
                        st = model.NewBoolVar(f"sstreak_{sname}_{dname}_{s}_{k}")
                        for p in parts:
                            model.Add(st <= p)
                        model.Add(st >= sum(parts) - k + 1)
                        if s_inc[k - 1] > 0:
                            student_streak_exprs[sname].append(
                                st
                                * s_inc[k - 1]
                                * student_importance[sname]
                                * student_size[sname]
                            )
    teacher_streak_exprs = {
        k: sum(v) if v else 0 for k, v in teacher_streak_exprs.items()
    }
    student_streak_exprs = {
        k: sum(v) if v else 0 for k, v in student_streak_exprs.items()
    }

    teacher_dup_exprs = {
        t: sum(v) if v else 0 for t, v in teacher_dup_exprs_map.items()
    }
    student_dup_exprs = {
        s: sum(v) if v else 0 for s, v in student_dup_exprs_map.items()
    }

    # penalties for consecutive days of the same subject
    consecutive_vars = []
    consecutive_map = {}
    for sid, subj in subjects.items():
        if not avoid_map.get(sid, True):
            continue
        count = len(subj["classes"])
        if count <= 1:
            continue
        # gather all (i, j) pairs once and reuse per class index
        pair_by_class: Dict[int, List[cp_model.IntVar]] = defaultdict(list)
        for i in range(count):
            for j in range(i + 1, count):
                var = model.NewBoolVar(f"cons_{sid}_{i}_{j}")
                model.Add(
                    class_day_idx[(sid, j)] == class_day_idx[(sid, i)] + 1
                ).OnlyEnforceIf(var)
                model.Add(
                    class_day_idx[(sid, j)] != class_day_idx[(sid, i)] + 1
                ).OnlyEnforceIf(var.Not())
                pair_by_class[i].append(var)
                pair_by_class[j].append(var)

        for idx in range(count):
            pv = pair_by_class.get(idx)
            if pv:
                any_v = model.NewBoolVar(f"cons_any_{sid}_{idx}")
                model.AddMaxEquality(any_v, pv)
                consecutive_vars.append(any_v)
                consecutive_map[(sid, idx)] = any_v

    # objective
    model_params = cfg.get("model", {})
    obj_mode = settings.get("objective", ["total"])[0]

    teacher_gap_exprs = {
        t: gap_teacher_val
        * teacher_importance[t]
        * sum(var for var, tt in teacher_gap_vars if tt == t)
        for t in teacher_names
    }

    student_gap_exprs = {
        s: gap_student_val
        * student_importance[s]
        * student_size[s]
        * sum(var for var, ss in student_gap_vars if ss == s)
        for s in student_importance
    }

    student_unopt_exprs = {
        s: sum(
            c["student_pen"].get(s, 0) * c["var"]
            for cand_list in candidates.values()
            for c in cand_list
        )
        for s in student_importance
    }

    opt_pen_val = penalties.get("optionalSubjectMissing", [0])[0]
    student_optional_exprs = {
        s: opt_pen_val
        * student_importance.get(s, default_student_imp)
        * student_size[s]
        * sum(optional_exprs_map.get(s, []))
        for s in student_size
    }

    teacher_unopt_exprs = {}
    for t in teacher_importance:
        expr = []
        for cand_list in candidates.values():
            for c in cand_list:
                pres = c.get("teacher_pres", {}).get(t)
                if pres is not None:
                    val = c["teacher_pen"].get(t, 0)
                    expr.append(val * pres)
        teacher_unopt_exprs[t] = sum(expr) if expr else 0

    teacher_consec_exprs = {}
    for t in teacher_importance:
        expr = []
        for (sid, j), var in consecutive_map.items():
            for cand in candidates[(sid, j)]:
                pres = cand.get("teacher_pres", {}).get(t)
                if pres is None:
                    continue
                both = model.NewBoolVar(
                    f"cons_t_{sid}_{j}_{t}_{cand['day']}_{cand['start']}"
                )
                model.AddMultiplicationEquality(both, [var, pres])
                expr.append(
                    both
                    * consecutive_pen_val
                    * teacher_importance[t]
                    * teacher_as_students
                )
        teacher_consec_exprs[t] = sum(expr) if expr else 0

    total_expr = (
            sum(teacher_gap_exprs.values())
            + sum(teacher_unopt_exprs.values())
            + sum(teacher_consec_exprs.values())
            + sum(teacher_streak_exprs.values())
            + sum(teacher_dup_exprs.values())
            + sum(student_gap_exprs.values())
            + sum(student_unopt_exprs.values())
            + sum(student_optional_exprs.values())
            + sum(student_streak_exprs.values())
            + sum(student_dup_exprs.values())
        )
    model.Minimize(total_expr)

    solver = cp_model.CpSolver()

    model_params = cfg.get("model", {})
    max_time = model_params.get("maxTime", DEFAULT_MAX_TIME)
    if isinstance(max_time, list):
        max_time = max_time[0]
    workers = model_params.get("workers", DEFAULT_WORKERS)
    if isinstance(workers, list):
        workers = workers[0] or DEFAULT_WORKERS
    show_progress = model_params.get("showProgress", DEFAULT_SHOW_PROGRESS)
    if isinstance(show_progress, list):
        show_progress = show_progress[0]
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = workers
    solver.parameters.log_search_progress = show_progress

    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        print("No feasible schedule found. Diagnosing configuration...")
        diagnose_config(cfg)
        raise RuntimeError("No feasible schedule found")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for (sid, idx), cand_list in candidates.items():
        for c in cand_list:
            if solver.Value(c["var"]):
                selected_cabs = [
                    cab
                    for cab, cv in c.get("cabinet_pres", {}).items()
                    if solver.Value(cv)
                ]
                assigned_teachers = [
                    t for t, tv in c.get("teacher_pres", {}).items() if solver.Value(tv)
                ]
                cls_obj = {
                    "subject": sid,
                    "teachers": assigned_teachers,
                    "cabinets": selected_cabs,
                    # copy list so modifications for optional classes don't affect others
                    "students": list(c["students"]),
                    "slotStudents": {
                        st: set(range(c["start"], c["start"] + c["length"]))
                        for st in c["students"]
                    },
                    "size": c["size"],
                    "optionalSize": 0,
                    "start": c["start"],
                    "length": c["length"],
                }
                for stu, slot_map in c.get("optional_pres", {}).items():
                    added = False
                    for sl, pres in slot_map.items():
                        if solver.Value(pres):
                            if not added:
                                cls_obj["students"].append(stu)
                                cls_obj.setdefault("slotStudents", {}).setdefault(stu, set())
                                cls_obj["optionalSize"] += student_size.get(stu, 1)
                                added = True
                            cls_obj["slotStudents"][stu].add(sl)
                for s in range(c["start"], c["start"] + c["length"]):
                    schedule[c["day"]][s].append(cls_obj)

    return schedule


def solve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate schedule and wrap it in export format."""
    schedule = build_model(cfg)
    optional_info = _collect_optional_stats(schedule, cfg)
    _deduplicate_schedule(schedule)

    teacher_names = [t["name"] for t in cfg.get("teachers", [])]
    student_names = [s["name"] for s in cfg.get("students", [])]
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}

    # configuration parameters referenced later in the function
    settings = cfg.get("settings", {})
    teacher_as_students = settings.get("teacherAsStudents", [15])[0]
    duplicates_pen_val = settings.get("duplicatesPenalty", [0])[0]

    penalties = cfg.get("penalties", {})
    teacher_streak_list = penalties.get("teacherLessonStreak", [[]])[0]
    student_streak_list = penalties.get("studentLessonStreak", [[]])[0]

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
                for t in cls.get("teachers", []):
                    teacher_slots[t][name].add(slot)
                for stu, slots_set in cls.get("slotStudents", {}).items():
                    if slot in slots_set:
                        student_slots[stu][name].add(slot)

    teacher_state = {t: {day["name"]: {} for day in cfg["days"]} for t in teacher_names}
    defaults = cfg.get("defaults", {})
    teach_arrive = {
        t["name"]: t.get(
            "arriveEarly",
            defaults.get("teacherArriveEarly", [False])[0],
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

    student_state = {s: {day["name"]: {} for day in cfg["days"]} for s in student_names}
    stud_arrive = {
        s["name"]: s.get(
            "arriveEarly",
            defaults.get("studentArriveEarly", [True])[0],
        )
        for s in cfg.get("students", [])
    }
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

    student_limits: Dict[str, Dict[str, Set[int]]] = {}
    for stu in cfg.get("students", []):
        name = stu["name"]
        allow = stu.get("allowedSlots")
        forbid = stu.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in cfg["days"]:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        student_limits[name] = avail

    penalties_cfg = {k: v[0] for k, v in cfg.get("penalties", {}).items()}
    def_teacher_imp = defaults.get("teacherImportance", [1])[0]
    def_student_imp = defaults.get("studentImportance", [0])[0]
    teacher_importance = {
        t["name"]: t.get("importance", def_teacher_imp) for t in cfg.get("teachers", [])
    }
    student_importance = {
        s["name"]: s.get("importance", def_student_imp) for s in cfg.get("students", [])
    }
    default_opt = defaults.get("optimalSlot", [0])[0]

    extra_pen = {} if duplicates_pen_val <= 0 else {"duplicate": 0}
    slot_penalties = {
        day["name"]: {
            slot: {**{k: 0 for k in penalties_cfg}, **extra_pen}
            for slot in day["slots"]
        }
        for day in cfg["days"]
    }
    slot_penalty_details = {
        day["name"]: {slot: [] for slot in day["slots"]} for day in cfg["days"]
    }

    # calculate penalties per slot using individual importance
    for day in cfg["days"]:
        dname = day["name"]
        teach_count = {t: 0 for t in teacher_names}
        stud_count = {s: 0 for s in student_names}
        for slot in day["slots"]:
            # gap penalties for teachers
            for t in teacher_names:
                if teacher_state[t][dname][slot] == "gap":
                    p = penalties_cfg.get("gapTeacher", 0) * teacher_importance[t]
                    slot_penalties[dname][slot]["gapTeacher"] += p
                    slot_penalty_details[dname][slot].append(
                        {"name": t, "type": "gapTeacher", "amount": p}
                    )
                if teacher_state[t][dname][slot] == "class":
                    teach_count[t] += 1
                    val_idx = (
                        min(len(teacher_streak_list) - 1, teach_count[t] - 1)
                        if teacher_streak_list
                        else -1
                    )
                    if val_idx >= 0:
                        pen = teacher_streak_list[val_idx]
                        if pen:
                            amount = pen * teacher_importance[t] * teacher_as_students
                            slot_penalties[dname][slot]["teacherLessonStreak"] += amount
                            slot_penalty_details[dname][slot].append(
                                {
                                    "name": t,
                                    "type": "teacherLessonStreak",
                                    "amount": amount,
                                }
                            )
                else:
                    teach_count[t] = 0
            # gap penalties for students
            for sname in student_names:
                if student_state[sname][dname][slot] == "gap":
                    p = (
                        penalties_cfg.get("gapStudent", 0)
                        * student_importance[sname]
                        * student_size.get(sname, 1)
                    )
                    slot_penalties[dname][slot]["gapStudent"] += p
                    slot_penalty_details[dname][slot].append(
                        {"name": sname, "type": "gapStudent", "amount": p}
                    )
                if student_state[sname][dname][slot] == "class":
                    stud_count[sname] += 1
                    val_idx = (
                        min(len(student_streak_list) - 1, stud_count[sname] - 1)
                        if student_streak_list
                        else -1
                    )
                    if val_idx >= 0:
                        pen = student_streak_list[val_idx]
                        if pen:
                            amount = (
                                pen
                                * student_importance[sname]
                                * student_size.get(sname, 1)
                            )
                            slot_penalties[dname][slot]["studentLessonStreak"] += amount
                            slot_penalty_details[dname][slot].append(
                                {
                                    "name": sname,
                                    "type": "studentLessonStreak",
                                    "amount": amount,
                                }
                            )
                else:
                    stud_count[sname] = 0

            if duplicates_pen_val > 0:
                classes = schedule[dname][slot]
                teach_cnt = defaultdict(int)
                stud_cnt = defaultdict(int)
                for cls in classes:
                    for t in cls.get("teachers", []):
                        teach_cnt[t] += 1
                    for s in cls["students"]:
                        stud_cnt[s] += 1
                for t, cnt in teach_cnt.items():
                    if cnt > 1:
                        p = (cnt - 1) * duplicates_pen_val * teacher_importance.get(t, def_teacher_imp)
                        slot_penalties[dname][slot]["duplicate"] += p
                        slot_penalty_details[dname][slot].append({"name": t, "type": "duplicate", "amount": p})
                for sname, cnt in stud_cnt.items():
                    if cnt > 1:
                        p = (cnt - 1) * duplicates_pen_val * student_importance[sname] * student_size.get(sname, 1)
                        slot_penalties[dname][slot]["duplicate"] += p
                        slot_penalty_details[dname][slot].append({"name": sname, "type": "duplicate", "amount": p})

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
                for t in cls.get("teachers", []):
                    p = (
                        base
                        * teacher_importance.get(t, def_teacher_imp)
                        * teacher_as_students
                    )
                    slot_penalties[dname][slot]["unoptimalSlot"] += p
                    slot_penalty_details[dname][slot].append(
                        {"name": t, "type": "unoptimalSlot", "amount": p}
                    )
                for sname in cls["students"]:
                    p = base * student_importance[sname] * student_size.get(sname, 1)
                    slot_penalties[dname][slot]["unoptimalSlot"] += p
                    slot_penalty_details[dname][slot].append(
                        {"name": sname, "type": "unoptimalSlot", "amount": p}
                    )

    # penalties for consecutive classes
    avoid_default = defaults.get("avoidConsecutive", [True])[0]
    avoid_map = {
        sid: cfg["subjects"][sid].get("avoidConsecutive", avoid_default)
        for sid in cfg["subjects"]
    }
    day_index = {day["name"]: idx for idx, day in enumerate(cfg["days"])}
    subj_occurrence = defaultdict(list)
    for day in cfg["days"]:
        dname = day["name"]
        d_idx = day_index[dname]
        for slot in day["slots"]:
            for cls in schedule[dname][slot]:
                if slot != cls["start"]:
                    continue
                sid = cls["subject"]
                subj_occurrence[sid].append((d_idx, dname, slot))
    for sid, occ in subj_occurrence.items():
        if not avoid_map.get(sid, True):
            continue
        occ.sort()
        for i in range(1, len(occ)):
            prev_idx = occ[i - 1][0]
            cur_idx, dname, slot = occ[i]
            if cur_idx == prev_idx + 1:
                base = penalties_cfg.get("consecutiveClass", 0)
                for cls in [c for c in schedule[dname][slot] if c["subject"] == sid]:
                    for t in cls.get("teachers", []):
                        p = (
                            base
                            * teacher_importance.get(t, def_teacher_imp)
                            * teacher_as_students
                        )
                        slot_penalties[dname][slot]["consecutiveClass"] += p
                        slot_penalty_details[dname][slot].append(
                            {"name": t, "type": "consecutiveClass", "amount": p}
                        )

    total_penalty = 0
    for day_map in slot_penalties.values():
        for pen in day_map.values():
            total_penalty += sum(pen.values())

    optional_stats: Dict[str, Dict[str, int | float]] = {}
    opt_pen_val = penalties_cfg.get("optionalSubjectMissing", 0)
    for stu in cfg.get("students", []):
        name = stu["name"]
        info = optional_info.get(name, {"total": 0, "attended": 0, "subjects": {}})
        missed_hours = info["total"] - info["attended"]
        penalty = (
            missed_hours
            * opt_pen_val
            * student_importance.get(name, def_student_imp)
            * student_size.get(name, 1)
        )
        total_penalty += penalty
        info["penalty"] = penalty
        optional_stats[name] = info

    export = {"days": []}
    for day in cfg["days"]:
        name = day["name"]
        slot_list = []
        for slot in day["slots"]:
            classes = []
            for cls in schedule[name][slot]:
                cp = cls.copy()
                if "slotStudents" in cp:
                    cp["slotStudents"] = {
                        k: sorted(list(v)) for k, v in cp["slotStudents"].items()
                    }
                classes.append(cp)
            gaps_students = [
                s for s in student_names if student_state[s][name][slot] == "gap"
            ]
            gaps_teachers = [
                t for t in teacher_names if teacher_state[t][name][slot] == "gap"
            ]
            home_students = [
                s for s in student_names if student_state[s][name][slot] == "home"
            ]
            home_teachers = [
                t for t in teacher_names if teacher_state[t][name][slot] == "home"
            ]
            slot_list.append(
                {
                    "slotIndex": slot,
                    "classes": classes,
                    "gaps": {"students": gaps_students, "teachers": gaps_teachers},
                    "home": {"students": home_students, "teachers": home_teachers},
                    "penalty": slot_penalties[name][slot],
                    "penaltyDetails": slot_penalty_details[name][slot],
                }
            )
        export["days"].append({"name": name, "slots": slot_list})
    export["totalPenalty"] = total_penalty
    export["optionalStats"] = optional_stats

    return export


def solve_fast(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Solve using the simplified model and compute total penalty."""
    schedule = build_fast_model(cfg)
    optional_info = _collect_optional_stats(schedule, cfg)
    _deduplicate_schedule(schedule)

    teacher_names = [t["name"] for t in cfg.get("teachers", [])]
    student_names = [s["name"] for s in cfg.get("students", [])]
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}

    teacher_last = {t: {d["name"]: -1 for d in cfg["days"]} for t in teacher_names}
    student_last = {s: {d["name"]: -1 for d in cfg["days"]} for s in student_names}
    student_slots_map = {s: {d["name"]: set() for d in cfg["days"]} for s in student_names}

    for day in cfg["days"]:
        dname = day["name"]
        for slot in day["slots"]:
            for cls in schedule[dname][slot]:
                end_slot = cls["start"] + cls["length"] - 1
                for t in cls.get("teachers", []):
                    teacher_last[t][dname] = max(teacher_last[t][dname], end_slot)
                for stu, slots_set in cls.get("slotStudents", {}).items():
                    if slot in slots_set:
                        student_last[stu][dname] = max(student_last[stu][dname], end_slot)
                        for s in range(cls["start"], end_slot + 1):
                            if s in slots_set:
                                student_slots_map[stu][dname].add(s)

    total_penalty = 0
    for t in teacher_names:
        for d in cfg["days"]:
            val = teacher_last[t][d["name"]]
            if val >= 0:
                total_penalty += val + 1
    for s in student_names:
        weight = student_size.get(s, 1)
        for d in cfg["days"]:
            val = student_last[s][d["name"]]
            if val >= 0:
                total_penalty += (val + 1) * weight

    optional_stats: Dict[str, Dict[str, int | float]] = {}
    opt_pen_val = cfg.get("penalties", {}).get("optionalSubjectMissing", [0])[0]
    def_imp = cfg.get("defaults", {}).get("studentImportance", [0])[0]
    importance = {s["name"]: s.get("importance", def_imp) for s in cfg.get("students", [])}
    for stu in cfg.get("students", []):
        name = stu["name"]
        info = optional_info.get(name, {"total": 0, "attended": 0, "subjects": {}})
        missed_hours = info["total"] - info["attended"]
        penalty = (
            missed_hours
            * opt_pen_val
            * importance.get(name, def_imp)
            * student_size.get(name, 1)
        )
        total_penalty += penalty
        info["penalty"] = penalty
        optional_stats[name] = info

    export = {"days": []}
    for day in cfg["days"]:
        dname = day["name"]
        slot_list = []
        for slot in day["slots"]:
            classes = []
            for cls in schedule[dname][slot]:
                cp = cls.copy()
                if "slotStudents" in cp:
                    cp["slotStudents"] = {k: sorted(list(v)) for k, v in cp["slotStudents"].items()}
                classes.append(cp)
            slot_list.append(
                {
                    "slotIndex": slot,
                    "classes": classes,
                    # placeholders for compatibility with generate_html
                    "gaps": {"students": [], "teachers": []},
                    "home": {"students": [], "teachers": []},
                    "penalty": {},
                    "penaltyDetails": [],
                }
            )
        export["days"].append({"name": dname, "slots": slot_list})
    export["totalPenalty"] = total_penalty
    export["optionalStats"] = optional_stats

    return export


def analyse_teachers(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """Gather statistics about teacher workload."""
    teachers = defaultdict(
        lambda: {"blocks": 0, "students": [], "subjects": defaultdict(list)}
    )
    for day in schedule.get("days", []):
        for slot in day.get("slots", []):
            for cls in slot.get("classes", []):
                sid = cls["subject"]
                count = cls.get("size", len(cls.get("students", []))) + cls.get("optionalSize", 0)
                for tid in cls.get("teachers", []):
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
                for st, slots_set in cls.get("slotStudents", {}).items():
                    if slot in slots_set:
                        student_hours[st][sid] += 1
    return student_hours


def analyse_subjects(schedule: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Collect information about subjects."""
    student_size = {s["name"]: int(s.get("group", 1)) for s in cfg.get("students", [])}
    subjects = defaultdict(
        lambda: {
            "students": set(),
            "student_count": 0,
            "teachers": set(),
            "class_sizes": [],
        }
    )
    for day in schedule.get("days", []):
        for slot in day.get("slots", []):
            for cls in slot.get("classes", []):
                sid = cls["subject"]
                for tid in cls.get("teachers", []):
                    subjects[sid]["teachers"].add(tid)
                subjects[sid]["class_sizes"].append(
                    cls.get("size", len(cls.get("students", [])))
                    + cls.get("optionalSize", 0)
                )
                for stu in cls.get("slotStudents", {}):
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
    return name[:idx], name[idx + 1 :]


def _format_table(
    rows, header_top=None, header_bottom=None, center_mask=None, join_rows=False
):
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

    for tid, info in sorted(
        teachers.items(), key=lambda x: teacher_hours(x[1]), reverse=True
    ):
        name1, name2 = _split_two_parts(teacher_names.get(tid, tid))
        total_hours = teacher_hours(info)
        row_top = [name1]
        row_bottom = [name2]
        subj_stats = sorted(
            info["subjects"].items(), key=lambda x: len(x[1]), reverse=True
        )
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
    for sid, subj_map in sorted(
        students.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        name = student_names.get(sid, sid)
        subj_count = len(subj_map)
        total_hours = sum(subj_map.values())
        parts = []
        for sub_id, hours in sorted(
            subj_map.items(), key=lambda x: subject_names.get(x[0], x[0])
        ):
            parts.append(f"{hours} hours {subject_names.get(sub_id, sub_id)}")
        line = (
            f"{name} has {subj_count} subjects for {total_hours} hours: "
            + ", ".join(parts)
        )
        lines.append(line)
    return "\n".join(lines)


def _subject_table(subjects, subject_names):
    rows = []
    for sid, info in sorted(
        subjects.items(), key=lambda x: subject_names.get(x[0], x[0])
    ):
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

    teacher_names = {
        t["name"]: t.get("printName", t.get("name", t["name"]))
        for t in cfg.get("teachers", [])
    }
    student_names = {
        s["name"]: s.get("name", s["name"]) for s in cfg.get("students", [])
    }
    subject_names = {
        sid: info.get("printName", info.get("name", sid))
        for sid, info in cfg.get("subjects", {}).items()
    }

    print("=== Teachers ===")
    print(_teacher_table(teachers, teacher_names, subject_names))

    print("\n=== Students ===")
    print(_student_list(students, student_names, subject_names))

    print("\n=== Subjects ===")
    print(_subject_table(subjects, subject_names))


def render_schedule(schedule: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Print schedule grouped by day and slot."""
    subject_names = {
        sid: info.get("printName", info.get("name", sid))
        for sid, info in cfg.get("subjects", {}).items()
    }
    teacher_names = {
        t["name"]: t.get("printName", t.get("name", t["name"]))
        for t in cfg.get("teachers", [])
    }
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

            header = (
                f"  Lesson {idx+1} [gap T:{gap_t} S:{gap_s} home T:{home_t} S:{home_s}]"
            )
            if not classes:
                print(f"{header}: --")
                continue
            print(f"{header}:")
            for cls in classes:
                subj = subject_names.get(cls["subject"], cls["subject"])
                teacher = ", ".join(
                    teacher_names.get(t, t) for t in cls.get("teachers", [])
                )
                base_size = cls.get("size", len(cls.get("students", [])))
                opt_size = cls.get("optionalSize", 0)
                size = base_size + opt_size
                length = cls.get("length", 1)
                start = cls.get("start", idx)
                part = ""
                if length > 1:
                    pno = idx - start + 1
                    part = f" (part {pno}/{length})"
                size_str = f"{base_size}" + (f" + {opt_size}" if opt_size else "")
                print(f"    {subj} by {teacher} for {size_str} students, {length}h{part}")
        print()


def generate_html(
    schedule: Dict[str, Any],
    cfg: Dict[str, Any],
    path: str = "schedule.html",
    generated: str | None = None,
    include_config: bool = False,
    raw_config: str | None = None,
) -> None:
    """Create interactive HTML overview of the schedule."""
    schedule_json = json.dumps(schedule, ensure_ascii=False)
    cfg_json = json.dumps(cfg, ensure_ascii=False)
    raw_json = json.dumps(raw_config) if raw_config is not None else "null"
    html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Schedule Overview</title>
<style>
body { font-family: Arial, sans-serif; }
.schedule-grid { border-collapse: collapse; width: 100%; display:grid; }
.schedule-grid .cell, .schedule-grid .header { border:1px solid #999; padding:4px; vertical-align:top; }
.schedule-grid .cell { display:flex; flex-direction:column; }
.schedule-grid .header { background:#f0f0f0; text-align:center; }
.mini-grid { margin-top:10px; }
.class-block { display:flex; flex-direction:column; margin-bottom:4px; }
.class-block.optional{background:#e0f7ff;}
.class-line { display:flex; gap:4px; width:100%; }
.class-line span { flex:1; }
.cls-subj { flex:0 0 50%; text-align:left; }
.cls-room { flex:0 0 30%; text-align:right; }
.cls-part { flex:0 0 20%; text-align:right; }
.slot-info { display:flex; gap:4px; justify-content:space-between; font-size:0.9em; color:#555; cursor:pointer; margin-top:auto; }
.slot-info span { flex:1 1 20%; text-align:center; }
.modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); }
.modal-content { background:#fff; position:fixed; top:5vh; bottom:5vh; left:10vw; right:10vw; margin:0; padding:20px; overflow:auto; }
.modal-header { position:relative; text-align:center; margin-bottom:10px; }
.close { position:absolute; right:0; top:0; cursor:pointer; font-size:20px; }
.history { display:inline-flex; gap:6px; justify-content:center; flex-wrap:wrap; }
.hist-item { font-size:0.9em; color:#888; cursor:pointer; }
.hist-item.active { font-weight:bold; color:#000; }
.clickable { color:#0066cc; cursor:pointer; text-decoration:underline; }
.sortable { cursor:pointer; }
.dup-teacher{background:orange;}
.dup-subject{background:orange;}
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
.overview-table{border:1px solid #999;border-collapse:collapse;width:70%;margin:0 auto;}
.overview-header,.overview-row{display:flex;align-items:center;}
.overview-header span,.overview-row span{border-right:1px solid #999;text-align:center;}
.overview-header span:last-child,.overview-row span:last-child{border-right:none;}
.overview-row{border-top:1px solid #999;}
.overview-row:first-child{border-top:none;}
.overview-header{background:#f0f0f0;font-weight:bold;}
.person-name{flex:0 0 25%;text-align:left;}
.person-info{flex:0 0 6%;text-align:center;}
.person-pen{flex:0 0 6%;text-align:center;}
.person-dup{flex:0 0 6%;text-align:center;}
.person-hours{flex:0 0 6%;text-align:center;}
.person-time{flex:0 0 6%;text-align:center;}
.person-opt{flex:0 0 6%;text-align:center;}
.person-load{flex:0 0 6%;text-align:center;}
.person-subjects{flex:0 0 45%;text-align:left;padding:0;}
.subject-list{display:flex;flex-direction:column;}
.subject-line{display:flex;gap:6px;border-top:1px solid #ddd;padding:2px 4px;}
.subject-line:first-child{border-top:none;}
.subject-name{flex:0 0 60%;text-align:left;}
.subject-count{flex:0 0 20%;text-align:center;}
.subject-extra{flex:0 0 20%;text-align:center;}
/* parameter blocks */
.param-table{--gap:8px;display:flex;flex-wrap:wrap;gap:var(--gap);justify-content:center;margin-top:10px;}
.param-block{flex:0 0 calc((100% - (6*var(--gap)))/6);border:1px solid #999;text-align:center;}
.param-block div{margin:0 0 5px 0; padding: 4px 0;}
.param-name{background:#f0f0f0;}
.meta{font-size:0.9em;margin-bottom:10px;}
pre{white-space:pre-wrap;word-break:break-all;background:#f8f8f8;padding:10px;}
</style>
</head>
<body>
<h1>Schedule Overview</h1>
__META__
<div id="table" class="schedule-grid"></div>
<h2 class="overview-section">Teachers Overview</h2>
<div id="teachers" class="overview-table"></div>
<h2 class="overview-section">Students Overview</h2>
<div id="students" class="overview-table"></div>
<div id="modal" class="modal"><div class="modal-content"><div class="modal-header"><div id="history" class="history"></div><span id="close" class="close">&#10006;</span></div><div id="modal-body"></div></div></div>
<script>
const scheduleData = __SCHEDULE__;
const configData = __CONFIG__;
const configRaw = __CONFIG_RAW__;
</script>
<script>
(function(){
const modal=document.getElementById('modal');
const close=document.getElementById('close');
const historyBox=document.getElementById('history');
const modalBody=document.getElementById('modal-body');
const cfgLink=document.getElementById('show-config');
close.onclick=()=>{modal.style.display='none';};
window.onclick=e=>{if(e.target==modal)modal.style.display='none';};
if(cfgLink){cfgLink.onclick=()=>{showConfig();};}
const studentSize={};
(configData.students||[]).forEach((s,i)=>{studentSize[s.name]=s.group||1;});
const teacherIndex={};
const teacherDisplay={};
(configData.teachers||[]).forEach((t,i)=>{teacherIndex[t.name]=i;teacherDisplay[t.name]=t.printName||t.name;});
const studentIndex={};
(configData.students||[]).forEach((s,i)=>{studentIndex[s.name]=i;});
const subjectDisplay={};
Object.keys(configData.subjects||{}).forEach(id=>{const info=configData.subjects[id]||{};subjectDisplay[id]=info.printName||info.name||id;});
const cabinetDisplay={};
Object.keys(configData.cabinets||{}).forEach(id=>{const info=configData.cabinets[id]||{};cabinetDisplay[id]=info.printName||id;});
const teacherSet=new Set(Object.keys(teacherIndex));
const studentSet=new Set(Object.keys(studentIndex));
function personLink(name,role){
  if(role==='teacher' || (!role && teacherSet.has(name) && !studentSet.has(name))){
    const id=teacherIndex[name];
    return '<span class="clickable teacher" data-id="'+id+'">'+(teacherDisplay[name]||name)+'</span>';
  }
  if(role==='student' || (!role && studentSet.has(name) && !teacherSet.has(name))){
    const id=studentIndex[name];
    return '<span class="clickable student" data-id="'+id+'">'+name+'</span>';
  }
  return name;
}
function teacherSpan(name,subj,dup){
  const id=teacherIndex[name];
  const prim=(configData.subjects[subj]||{}).primaryTeachers||[];
  const disp=teacherDisplay[name]||name;
  const inner=prim.includes(name)?'<strong>'+disp+'</strong>':disp;
  const cls='clickable teacher'+(dup?' dup-teacher':'');
  return '<span class="'+cls+'" data-id="'+id+'">'+inner+'</span>';
}
function cabinetSpan(name){
  return '<span class="clickable cabinet" data-id="'+name+'">'+(cabinetDisplay[name]||name)+'</span>';
}
function makeParamTable(list){
  let html='<div class="param-table">';
  list.forEach(item=>{
    const name = Array.isArray(item) ? item[0] : item.name;
    const val = Array.isArray(item) ? item[1] : item.value;
    const bold = Array.isArray(item) ? item[2] : item.bold;
    const valueHtml = bold ? '<strong>'+val+'</strong>' : val;
    html += '<div class="param-block"><div class="param-name">'+name+'</div><div>'+valueHtml+'</div></div>';
  });
  html+='</div>';return html;
}
function countStudents(list){return(list||[]).reduce((a,n)=>a+(studentSize[n]||1),0);}
function studentsAt(cls,slot){
  if(!cls.slotStudents) return cls.students||[];
  const res=[];const seen=new Set();
  (cls.students||[]).forEach(n=>{if(!cls.slotStudents[n]||cls.slotStudents[n].includes(slot)){res.push(n);seen.add(n);}});
  Object.keys(cls.slotStudents).forEach(n=>{if(!seen.has(n)&&cls.slotStudents[n].includes(slot))res.push(n);});
  return res;
}
function studentPresent(cls,name,slot){
  if(!cls.slotStudents||!cls.slotStudents[name]) return (cls.students||[]).includes(name);
  return cls.slotStudents[name].includes(slot);
}
let historyStack=[];
let historyTitles=[];
let historyIndex=-1;
function getTitle(html){const m=html.match(/<h2[^>]*>(.*?)<\/h2>/i);return m?m[1]:'';}
function renderModal(){
 modalBody.innerHTML=historyStack[historyIndex]||'';
 modal.style.display='block';
 historyBox.innerHTML='';
 const start=Math.max(0,historyTitles.length-3);
 for(let i=start;i<historyTitles.length;i++){
   const item=document.createElement('span');
   item.className='hist-item'+(i===historyIndex?' active':'');
   item.textContent=historyTitles[i];
   item.onclick=()=>{historyIndex=i;renderModal();};
   historyBox.appendChild(item);
   if(i<historyTitles.length-1){
     const sep=document.createElement('span');
     sep.textContent='>';
     historyBox.appendChild(sep);
   }
 }
}
function openModal(html,reset=true){
 const title=getTitle(html);
 if(reset){
   historyStack=[html];
   historyTitles=[title];
 }else{
   const idx=historyTitles.indexOf(title);
   if(idx>=0){historyStack.splice(idx,1);historyTitles.splice(idx,1);}
   historyStack.push(html);historyTitles.push(title);
   if(historyStack.length>3){
     historyStack=historyStack.slice(-3);
     historyTitles=historyTitles.slice(-3);
   }
}
historyIndex=historyStack.length-1;
renderModal();
}
function showConfig(){
 const html='<h2>Configuration</h2><button id="cfg-copy">Copy</button><pre id="cfg-pre"></pre>';
 openModal(html);
 document.getElementById('cfg-pre').textContent=configRaw;
 const btn=document.getElementById('cfg-copy');
 if(btn){btn.onclick=()=>{navigator.clipboard.writeText(configRaw);};}
}
const COLOR_MIN=[220,255,220];
const COLOR_MID=[255,255,255];
const COLOR_MAX=[255,220,220];
const teacherSort={field:'penalty',asc:false};
const studentSort={field:'penalty',asc:false};
const duplicatesAllowed=((configData.settings||{}).duplicatesPenalty||[0])[0]>0;

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
   const hdr=document.createElement('div');hdr.className='header';hdr.textContent='Lesson '+(i+1);container.appendChild(hdr);
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i) || {classes:[],gaps:{students:[],teachers:[]},home:{students:[],teachers:[]},penalty:{}};
     const cell=document.createElement('div');
     cell.className='cell';
     const pVal=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
     minP=Math.min(minP,pVal);maxP=Math.max(maxP,pVal);
    const tDup={};
    const sDup={};
    slot.classes.forEach(c=>{
      (c.teachers||[]).forEach(t=>{tDup[t]=(tDup[t]||0)+1;});
      studentsAt(c,i).forEach(s=>{sDup[s]=(sDup[s]||0)+1;});
    });
    slot.classes.forEach(cls=>{
      const block=document.createElement('div');
      block.className='class-block';
      const subj=subjectDisplay[cls.subject]||cls.subject;
      const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
      const l1=document.createElement('div');
      l1.className='class-line';
      const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
      const subjCls='cls-subj clickable subject'+(studentsAt(cls,i).some(s=>sDup[s]>1)?' dup-subject':'');
      l1.innerHTML='<span class="'+subjCls+'" data-id="'+cls.subject+'">'+subj+'</span>'+
        '<span class="cls-room">'+rooms+'</span>'+
        '<span class="cls-part">'+part+'</span>';
      const l2=document.createElement('div');
      l2.className='class-line';
      const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject,tDup[t]>1)).join(', ');
      const sizeStr=cls.optionalSize?cls.size+' + '+cls.optionalSize:cls.size; 
      l2.innerHTML='<span class="cls-teach">'+tNames+'</span>'+
       '<span class="cls-size">'+sizeStr+'</span>'; 
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

function makeGrid(filterFn, highlightFn){
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 let html='<div class="schedule-grid mini-grid" style="grid-template-columns:auto repeat('+scheduleData.days.length+',1fr)">';
 html+='<div></div>';
 scheduleData.days.forEach(d=>{html+='<div class="header">'+d.name+'</div>';});
 for(let i=0;i<maxSlots;i++){
  html+='<div class="header">Lesson '+(i+1)+'</div>';
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i)||{classes:[]};
     html+='<div class="cell">';
    const tDup={};
    const sDup={};
    slot.classes.forEach(c=>{(c.teachers||[]).forEach(t=>{tDup[t]=(tDup[t]||0)+1;});studentsAt(c,i).forEach(s=>{sDup[s]=(sDup[s]||0)+1;});});
    slot.classes.filter(c=>filterFn(c,i)).forEach(cls=>{
       const optional = highlightFn && highlightFn(cls);
       const subj=subjectDisplay[cls.subject]||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject,tDup[t]>1)).join(', ');
        const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
        const subjCls='cls-subj clickable subject'+(studentsAt(cls,i).some(s=>sDup[s]>1)?' dup-subject':'');
        const blockCls='class-block'+(optional?' optional':'');
        html+='<div class="'+blockCls+'">'+
         '<div class="class-line">'+
          '<span class="'+subjCls+'" data-id="'+cls.subject+'">'+subj+'</span>'+
          '<span class="cls-room">'+rooms+'</span>'+
          '<span class="cls-part">'+part+'</span>'+
        '</div>'+
        '<div class="class-line">'+
          '<span class="cls-teach">'+tNames+'</span>'+
          '<span class="cls-size">'+(cls.optionalSize?cls.size+' + '+cls.optionalSize:cls.size)+'</span>'+
        '</div>'+
       '</div>';
     });
     html+='</div>';
   });
 }
 html+='</div>';
 return html;
}

function showSlot(day,idx,fromModal=false){
 const d=scheduleData.days.find(x=>x.name===day);if(!d)return;
 const slot=d.slots.find(s=>s.slotIndex==idx);if(!slot)return;
 const total=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
 let html='<h2>'+day+' lesson '+(idx+1)+'</h2><p>Total penalty: '+total.toFixed(1)+'</p>';
 html+='<div class="slot-detail">';
 const tDup={};
 const sDup={};
slot.classes.forEach(c=>{(c.teachers||[]).forEach(t=>{tDup[t]=(tDup[t]||0)+1;});studentsAt(c,idx).forEach(s=>{sDup[s]=(sDup[s]||0)+1;});});
 slot.classes.forEach((cls)=>{
   const subj=subjectDisplay[cls.subject]||cls.subject;
   const part=(cls.length>1)?((idx-cls.start+1)+'/'+cls.length):'1/1';
  const subjCls='detail-subj clickable subject'+(studentsAt(cls,idx).some(s=>sDup[s]>1)?' dup-subject':'');
   html+='<div class="slot-class">'+
     '<div class="detail-line">'+
       '<span class="'+subjCls+'" data-id="'+cls.subject+'">'+subj+'</span>'+
      '<span class="detail-teacher">'+(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject,tDup[t]>1)).join(', ')+'</span>'+
       '<span class="detail-room">'+(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ')+'</span>'+
       '<span class="detail-size">'+(cls.optionalSize?cls.size+' + '+cls.optionalSize:cls.size)+'</span>'+
       '<span class="detail-part">'+part+'</span>'+
     '</div>';
  const studs=studentsAt(cls,idx).map(n=>personLink(n,'student')).join(', ');
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
        const isTeach=p.type==='gapTeacher'||(p.type==='unoptimalSlot'&&teacherSet.has(p.name))||(p.type==='consecutiveClass'&&teacherSet.has(p.name))||(p.type==='duplicate'&&teacherSet.has(p.name));
        const role=isTeach?'teacher':'student';
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
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let sizes=[],total=0,gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>(c.teachers||[]).includes(name)));
   if(teachSlots.length){
     const firstClass=teachSlots[0].slotIndex;
     const first=arrive?dayStart:firstClass;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     teachSlots.forEach(sl=>{const c=sl.classes.find(x=>(x.teachers||[]).includes(name));sizes.push(c.size+(c.optionalSize||0));total++;});
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
   }
  });
 const avg=sizes.reduce((a,b)=>a+b,0)/(sizes.length||1);
 return{totalClasses:total,avgSize:avg.toFixed(1),gap:gap,time:time};
}

function computeStudentStats(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>studentPresent(c,name,sl.slotIndex)));
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
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const defImp=(configData.defaults.teacherImportance||[1])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 const imp=info.importance!==undefined?info.importance:defImp;
 const allow=info.allowedSlots||null;
 const forbid=info.forbiddenSlots||null;
 let hours=0,gap=0,time=0,subjects={},pen=0,avail=0,dup=0;
  scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>(c.teachers||[]).includes(name)));
   if(teachSlots.length){
     const first=arrive?dayStart:teachSlots[0].slotIndex;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
     teachSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>(c.teachers||[]).includes(name));
       hours++;
       const stat=subjects[cls.subject]||{count:0,size:0};
       stat.count++;stat.size+=cls.size+(cls.optionalSize||0);subjects[cls.subject]=stat;
     });
   }
  slots.forEach(sl=>{
    const cnt=sl.classes.filter(c=>(c.teachers||[]).includes(name)).length;
    if(cnt>1)dup+=cnt-1;
    (sl.penaltyDetails||[]).forEach(p=>{if(p.name===name)pen+=p.amount;});
  });
  const allSlots=slots.map(s=>s.slotIndex);
  let allowed=new Set(allSlots);
  if(allow!==null){
    if(Object.prototype.hasOwnProperty.call(allow,day.name)){
      const arr=allow[day.name];
      allowed=new Set(arr.length?arr:allSlots);
    }else{
      allowed=new Set();
    }
  }
  if(forbid!==null && Object.prototype.hasOwnProperty.call(forbid,day.name)){
    const fb=forbid[day.name];
    if(!fb.length){
      allowed=new Set();
    }else{
      fb.forEach(x=>allowed.delete(x));
    }
  }
  avail+=allowed.size;
  });
  for(const k in subjects){subjects[k].avg=(subjects[k].size/subjects[k].count).toFixed(1);}
 const load=avail?hours*100/avail:0;
 const penVal=imp?pen/imp:0;
 return{arrive,imp,penalty:penVal,hours:hours,time:time,load:load,subjects,duplicates:dup};
}

function computeStudentInfo(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const defImp=(configData.defaults.studentImportance||[0])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 const imp=info.importance!==undefined?info.importance:defImp;
 let hours=0,gap=0,time=0,subjects={},pen=0,dup=0;
 const optStats=(scheduleData.optionalStats||{})[name]||{total:0,attended:0,penalty:0,subjects:{}};
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>studentPresent(c,name,sl.slotIndex)));
   if(stSlots.length){
     const first=arrive?dayStart:stSlots[0].slotIndex;
     const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
     stSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>studentPresent(c,name,sl.slotIndex));
       hours++;
       const stat=subjects[cls.subject]||{count:0,penalty:0};
       stat.count++;subjects[cls.subject]=stat;
     });
   }
   slots.forEach(sl=>{
    const cnt=sl.classes.filter(c=>studentPresent(c,name,sl.slotIndex)).length;
    if(cnt>1)dup+=cnt-1;
    (sl.penaltyDetails||[]).forEach(p=>{
    if(p.name===name){
      pen+=p.amount;
      if(p.type==='unoptimalSlot'){
        const cls=sl.classes.find(c=>studentPresent(c,name,sl.slotIndex));
        if(cls){
          subjects[cls.subject]=subjects[cls.subject]||{count:0,penalty:0};
          subjects[cls.subject].penalty+=(p.amount/imp);
        }
      }
    }
  });});
});
pen+=optStats.penalty;
const penVal=imp?pen/imp:0;
return{arrive,imp,penalty:penVal,hours:hours,time:time,subjects,duplicates:dup,optGot:optStats.attended,optMiss:optStats.total,optDetail:optStats.subjects||{}};
}

function buildTeachers(){
 const cont=document.getElementById('teachers');
 cont.innerHTML='';
 const header=document.createElement('div');
  header.className='overview-header';
  let hHtml='<span class="person-name">Teacher</span>'+
    '<span class="person-info sortable" data-sort="priority">Priority<br>Arrive</span>'+
    '<span class="person-pen sortable" data-sort="penalty">Penalty</span>';
  if(duplicatesAllowed){
    hHtml+='<span class="person-dup sortable" data-sort="duplicates">Dup</span>';
  }
  hHtml+=
    '<span class="person-hours sortable" data-sort="hours">Hours</span>'+
    '<span class="person-time sortable" data-sort="time">At school</span>'+
    '<span class="person-load sortable" data-sort="load">Load %</span>'+
    '<span class="person-subjects">Subject<br>Cls | Avg</span>';
  header.innerHTML=hHtml;
  cont.appendChild(header);
  header.querySelectorAll('.sortable').forEach(el=>{
    el.onclick=()=>{
      const field=el.dataset.sort;
      if(teacherSort.field===field){teacherSort.asc=!teacherSort.asc;}else{teacherSort.field=field;teacherSort.asc=true;}
      buildTeachers();
    };
  });
  const infos=(configData.teachers||[]).map(t=>{return{info:t,stat:computeTeacherInfo(t.name)}});
 infos.sort((a,b)=>{
   let av, bv;
   switch(teacherSort.field){
     case 'priority': av=a.stat.imp; bv=b.stat.imp; break;
     case 'hours': av=a.stat.hours; bv=b.stat.hours; break;
     case 'time': av=a.stat.time; bv=b.stat.time; break;
     case 'load': av=a.stat.load; bv=b.stat.load; break;
     case 'duplicates': av=a.stat.duplicates; bv=b.stat.duplicates; break;
     default: av=a.stat.penalty; bv=b.stat.penalty;
   }
   if(av<bv) return teacherSort.asc?-1:1;
   if(av>bv) return teacherSort.asc?1:-1;
   return 0;
 });
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
  const arr=item.stat.arrive?"yes":"no";
  const pr=item.stat.imp;
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
    const name=subjectDisplay[sid]||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+s.avg+'</span></div>';
   });
  let rHtml='<span class="person-name clickable teacher" data-id="'+teacherIndex[item.info.name]+'">'+(teacherDisplay[item.info.name]||item.info.name)+'</span>'+
    '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
    '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>';
  if(duplicatesAllowed){
    rHtml+='<span class="person-dup">'+item.stat.duplicates+'</span>';
  }
  rHtml+=
    '<span class="person-hours">'+item.stat.hours+'</span>'+
    '<span class="person-time">'+item.stat.time+'</span>'+
    '<span class="person-load">'+item.stat.load.toFixed(1)+'</span>'+
    '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
  row.innerHTML=rHtml;
   cont.appendChild(row);
  });
}

function buildStudents(){
 const cont=document.getElementById('students');
 cont.innerHTML='';
 const header=document.createElement('div');
  header.className='overview-header';
  let hHtml='<span class="person-name">Student</span>'+
    '<span class="person-info sortable" data-sort="priority">Priority<br>Arrive</span>'+
    '<span class="person-pen sortable" data-sort="penalty">Penalty</span>';
  if(duplicatesAllowed){
    hHtml+='<span class="person-dup sortable" data-sort="duplicates">Dup</span>';
  }
  hHtml+=
    '<span class="person-hours sortable" data-sort="hours">Hours</span>'+
    '<span class="person-time sortable" data-sort="time">At school</span>'+
    '<span class="person-opt">Optional subjects</span>'+
    '<span class="person-subjects">Subject<br>Cls | Pen</span>';
  header.innerHTML=hHtml;
  cont.appendChild(header);
  header.querySelectorAll('.sortable').forEach(el=>{
    el.onclick=()=>{
      const field=el.dataset.sort;
      if(studentSort.field===field){studentSort.asc=!studentSort.asc;}else{studentSort.field=field;studentSort.asc=true;}
      buildStudents();
    };
  });
  const infos=(configData.students||[]).map(s=>{return{info:s,stat:computeStudentInfo(s.name)}});
infos.sort((a,b)=>{
  let av,bv;
  switch(studentSort.field){
    case 'priority': av=a.stat.imp; bv=b.stat.imp; break;
    case 'hours': av=a.stat.hours; bv=b.stat.hours; break;
    case 'time': av=a.stat.time; bv=b.stat.time; break;
    case 'duplicates': av=a.stat.duplicates; bv=b.stat.duplicates; break;
    default: av=a.stat.penalty; bv=b.stat.penalty;
  }
   if(av<bv) return studentSort.asc?-1:1;
   if(av>bv) return studentSort.asc?1:-1;
   return 0;
 });
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
  const arr=item.stat.arrive?"yes":"no";
  const pr=item.stat.imp;
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
    const name=subjectDisplay[sid]||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+(s.penalty||0).toFixed(1)+'</span></div>';
   });
  row.innerHTML='<span class="person-name clickable student" data-id="'+studentIndex[item.info.name]+'">'+item.info.name+'</span>'+
    '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
    '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>';
  if(duplicatesAllowed){
    row.innerHTML+= '<span class="person-dup">'+item.stat.duplicates+'</span>';
  }
  row.innerHTML+=
    '<span class="person-hours">'+item.stat.hours+'</span>'+
    '<span class="person-time">'+item.stat.time+'</span>'+
    '<span class="person-opt">'+item.stat.optGot+'/'+item.stat.optMiss+'</span>'+
    '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
   cont.appendChild(row);
  });
}

function showTeacher(idx,fromModal=false){
 const info=(configData.teachers||[])[idx]||{};
 const name=info.name||'';
 const display=teacherDisplay[name]||name;
 const defImp=(configData.defaults.teacherImportance||[1])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const boldImp=imp!==defImp;
 const stats=computeTeacherStats(name);
 const full=computeTeacherInfo(name);
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const boldArr=full.arrive!==defArr;
 let html='<h2>Teacher: '+display+'</h2>'+makeGrid((cls,_slot)=>cls.teachers.includes(name));
 html+='<h3>Subjects</h3><table class="info-table"><tr><th>Subject</th><th>Classes</th><th>Avg size</th></tr>';
 Object.keys(full.subjects).forEach(sid=>{
   const s=full.subjects[sid];
   const sname=subjectDisplay[sid]||sid;
   html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+sname+'</span></td><td class="num">'+s.count+'</td><td class="num">'+s.avg+'</td></tr>';});
 html+='</table>';
  const params=[
   ['Priority',imp,boldImp],
   ['Arrive early',full.arrive?'yes':'no',boldArr],
   ['Gap hours',stats.gap],
   ['At school',stats.time],
   ['Load %',full.load.toFixed(1)],
   ['Total classes',stats.totalClasses],
   ['Average size',stats.avgSize],
   ['Penalty',full.penalty.toFixed(1)]
  ];
  if(duplicatesAllowed){
    params.splice(params.length-1,0,['Duplicates',full.duplicates]);
  }
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showStudent(idx,fromModal=false){
 const info=(configData.students||[])[idx]||{};
 const name=info.name||'';
 const defImp=(configData.defaults.studentImportance||[0])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const boldImp=imp!==defImp;
 const group=studentSize[name]||1;
 const boldGroup=group!==1;
 const stats=computeStudentStats(name);
 const full=computeStudentInfo(name);
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const boldArr=full.arrive!==defArr;
 const optionalSubjects=info.optionalSubjects||[];
 let html='<h2>Student: '+name+'</h2>'+makeGrid((cls,slot)=>studentPresent(cls,name,slot),cls=>optionalSubjects.includes(cls.subject));
 html+='<h3>Subjects</h3><table class="info-table"><tr><th>Subject</th><th>Classes</th><th>Penalty</th></tr>';
Object.keys(full.subjects).forEach(sid=>{const s=full.subjects[sid];const sn=subjectDisplay[sid]||sid;html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></td><td class="num">'+s.count+'</td><td class="num">'+(s.penalty||0).toFixed(1)+'</td></tr>';});
 html+='</table>';
 if(optionalSubjects.length){
   html+='<h3>Optional subjects</h3><table class="info-table"><tr><th>Subject</th><th>Hours</th></tr>';
   optionalSubjects.forEach(sid=>{
     const st=full.optDetail[sid]||{attended:0,total:0};
     const nameDisp=subjectDisplay[sid]||sid;
     html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+nameDisp+'</span></td><td class="num">'+st.attended+'/'+st.total+'</td></tr>';
   });
   html+='</table>';
 }
  const params=[
   ['Group size',group,boldGroup],
   ['Priority',imp,boldImp],
   ['Arrive early',full.arrive?'yes':'no',boldArr],
   ['Gap hours',stats.gap],
   ['At school',stats.time],
   ['Penalty',full.penalty.toFixed(1)]
  ];
  if(duplicatesAllowed){
    params.splice(params.length-1,0,['Duplicates',full.duplicates]);
  }
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showCabinet(name,fromModal=false){
 const info=configData.cabinets[name]||{};
  const disp=cabinetDisplay[name]||name;
  let html='<h2>Room: '+disp+'</h2>'+makeGrid((cls,_slot)=>(cls.cabinets||[]).includes(name));
 const params=[
  ['Capacity',info.capacity||'-'],
  ['Allowed subjects',(info.allowedSubjects||[]).map(s=>subjectDisplay[s]||s).join(', ')||'-']
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showSubject(id,fromModal=false){
 const subj=configData.subjects[id]||{};
 const defOpt=(configData.defaults.optimalSlot||[0])[0];
 const disp=subjectDisplay[id]||id;
 let html='<h2>Subject: '+disp+'</h2>'+makeGrid((cls)=>cls.subject===id);
html+='<h3>Teachers</h3><table class="info-table"><tr><th>Name</th></tr>';
(configData.teachers||[]).forEach((t,i)=>{if((t.subjects||[]).includes(id)){const bold=(subj.primaryTeachers||[]).includes(t.name);const nm=bold?'<strong>'+(teacherDisplay[t.name]||t.name)+'</strong>':(teacherDisplay[t.name]||t.name);html+='<tr><td><span class="clickable teacher" data-id="'+i+'">'+nm+'</span></td></tr>';}});
 html+='</table><h3>Students</h3><table class="info-table"><tr><th>Name</th><th>Group</th></tr>';
 (configData.students||[]).forEach((s,i)=>{if((s.subjects||[]).includes(id)){html+='<tr><td><span class="clickable student" data-id="'+i+'">'+s.name+'</span></td><td class="num">'+(studentSize[s.name]||1)+'</td></tr>';}});
 html+='</table>';
 const defPerm=(configData.defaults.permutations||[true])[0];
 const defAvoid=(configData.defaults.avoidConsecutive||[true])[0];
 const opt=subj.optimalSlot!==undefined?subj.optimalSlot:defOpt;
 const boldOpt=opt!==defOpt;
 const perm=subj.allowPermutations!==undefined?subj.allowPermutations:defPerm;
 const boldPerm=perm!==defPerm;
 const avoid=subj.avoidConsecutive!==undefined?subj.avoidConsecutive:defAvoid;
 const boldAvoid=avoid!==defAvoid;
  const req=subj.requiredTeachers!==undefined?subj.requiredTeachers:1;
  const boldReq=req!==1;
  const reqCab=subj.requiredCabinets!==undefined?subj.requiredCabinets:1;
  const boldReqCab=reqCab!==1;
  const params=[
  ['Classes',(subj.classes||[]).join(', ')||'-'],
  ['Optimal lesson',opt+1,boldOpt],
  ['Allow permutations',perm?'yes':'no',boldPerm],
  ['Avoid consecutive',avoid?'yes':'no',boldAvoid],
   ['Required teachers',req,boldReq],
   ['Required cabinets',reqCab,boldReqCab],
   ['Cabinets',(subj.cabinets||[]).map(c=>cabinetDisplay[c]||c).join(', ')||'-'],
  ['Primary teachers',(subj.primaryTeachers||[]).map(t=>teacherDisplay[t]||t).join(', ')||'-']
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

document.addEventListener('click',e=>{
 const fromModal=modal.contains(e.target);
 const slotElem=e.target.closest('.slot-info');
 if(slotElem){
   showSlot(slotElem.dataset.day,parseInt(slotElem.dataset.slot),fromModal);
   return;
 }
 const target=e.target.closest('.subject,.teacher,.student,.cabinet');
 if(!target)return;
 if(target.classList.contains('subject')){showSubject(target.dataset.id,fromModal);}
 else if(target.classList.contains('teacher')){showTeacher(parseInt(target.dataset.id),fromModal);}
 else if(target.classList.contains('student')){showStudent(parseInt(target.dataset.id),fromModal);}
 else if(target.classList.contains('cabinet')){showCabinet(target.dataset.id,fromModal);}
});

buildTable();
buildTeachers();
buildStudents();
})();
</script>
</body>
</html>
"""
    meta = ""
    if generated:
        meta = f'<div class="meta">Generated: {generated}'
        if include_config:
            meta += ' | <span id="show-config" class="clickable">View config</span>'
        meta += "</div>"
    html = (
        html.replace("__SCHEDULE__", schedule_json)
        .replace("__CONFIG__", cfg_json)
        .replace("__CONFIG_RAW__", raw_json)
        .replace("__META__", meta)
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def diagnose_config(cfg: Dict[str, Any]) -> None:
    """Print possible bottlenecks for finding a feasible schedule."""
    days = cfg.get("days", [])
    subjects = cfg.get("subjects", {})
    teachers = cfg.get("teachers", [])
    students = cfg.get("students", [])

    # allowed slots for teachers
    teacher_limits: Dict[str, Dict[str, Set[int]]] = {}
    for t in teachers:
        name = t["name"]
        allow = t.get("allowedSlots")
        forbid = t.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        teacher_limits[name] = avail

    # allowed slots for students
    student_limits: Dict[str, Dict[str, Set[int]]] = {}
    for stu in students:
        name = stu["name"]
        allow = stu.get("allowedSlots")
        forbid = stu.get("forbiddenSlots")
        avail: Dict[str, Set[int]] = {}
        for day in days:
            dname = day["name"]
            slots_set = set(day["slots"])
            if allow is not None:
                if dname in allow:
                    al = allow[dname]
                    allowed = slots_set.copy() if not al else set(al)
                else:
                    allowed = set()
            else:
                allowed = slots_set.copy()
            if forbid is not None and dname in forbid:
                fb = forbid[dname]
                if not fb:
                    allowed = set()
                else:
                    allowed -= set(fb)
            avail[dname] = allowed
        student_limits[name] = avail

    teacher_hours_avail = {
        t: sum(len(s) for s in day_map.values()) for t, day_map in teacher_limits.items()
    }
    student_hours_avail = {
        s: sum(len(sl) for sl in day_map.values()) for s, day_map in student_limits.items()
    }
    teacher_map = {t["name"]: set(t.get("subjects", [])) for t in teachers}
    subject_hours = {sid: sum(info.get("classes", [])) for sid, info in subjects.items()}

    warnings = []

    for sid, info in subjects.items():
        required = int(info.get("requiredTeachers", 1))
        t_list = [t for t in teacher_map if sid in teacher_map[t]]
        if len(t_list) < required:
            warnings.append(
                f"Subject {sid} requires {required} teachers but only {len(t_list)} available."
            )

    teacher_hours_req = {t["name"]: 0.0 for t in teachers}
    for sid, info in subjects.items():
        hours = subject_hours.get(sid, 0) * int(info.get("requiredTeachers", 1))
        t_list = [t for t in teacher_map if sid in teacher_map[t]]
        if not t_list:
            continue
        share = hours / len(t_list)
        for t in t_list:
            teacher_hours_req[t] += share

    for t in teachers:
        name = t["name"]
        req = teacher_hours_req[name]
        avail = teacher_hours_avail.get(name, 0)
        if req > avail:
            warnings.append(
                f"Teacher {name} may need ~{req:.1f} hours but only {avail} allowed."
            )

    for stu in students:
        name = stu["name"]
        need = sum(subject_hours.get(sid, 0) for sid in stu.get("subjects", []))
        avail = student_hours_avail.get(name, 0)
        if need > avail:
            warnings.append(
                f"Student {name} needs {need} hours but only {avail} allowed."
            )

    if warnings:
        print("Potential issues:")
        for w in warnings:
            print(f" - {w}")
    else:
        print("No obvious issues detected.")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-y"]
    auto_yes = "-y" in sys.argv[1:]

    cfg_path = args[0] if len(args) >= 1 else "schedule-config.json"
    out_path = args[1] if len(args) >= 2 else "schedule.json"
    html_path = args[2] if len(args) >= 3 else "schedule.html"

    if not os.path.exists(cfg_path):
        print(f"Config '{cfg_path}' not found.")
        return

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw_cfg_text = fh.read()

    cfg = load_config(cfg_path)
    validate_config(cfg)

    obj_mode = cfg.get("settings", {}).get("objective", ["total"])[0]
    skip_solve = False
    if os.path.exists(out_path):
        if auto_yes:
            skip_solve = True
        else:
            ans = input(
                f"Schedule file '{out_path}' found. Skip solving and use it? [y/N] "
            )
            skip_solve = ans.strip().lower().startswith("y")

    student_dups = _detect_duplicates(cfg.get("students", []), ["subjects"])
    if student_dups and not skip_solve:
        print("Duplicate entities detected:")
        for names in student_dups:
            print(f"Students: {', '.join(names)}")
        if not auto_yes:
            ans = input("Continue despite duplicates? [y/N] ")
            if not ans.strip().lower().startswith("y"):
                print("Exiting due to duplicates.")
                return


    fresh = not skip_solve
    if skip_solve:
        with open(out_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
    else:
        if obj_mode == "fast":
            result = solve_fast(cfg)
        else:
            result = solve(cfg)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"Schedule written to {out_path}")

    render_schedule(result, cfg)

    if auto_yes:
        show_analysis = True
    else:
        ans = input("Show analysis and generate html page? [y/N] ")
        show_analysis = ans.strip().lower().startswith("y")

    if show_analysis:
        report_analysis(result, cfg)
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        generate_html(result, cfg, html_path, gen_time, fresh, raw_cfg_text)
        print(f"{html_path} generated")


if __name__ == "__main__":
    main()
