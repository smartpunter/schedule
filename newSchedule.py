import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Any, Set
from ortools.sat.python import cp_model

DEFAULT_MAX_TIME = 10800  # 3 hours
DEFAULT_SHOW_PROGRESS = True
DEFAULT_WORKERS = max(os.cpu_count() - 2, 4) if os.cpu_count() else 4


def _detect_duplicates(entities: List[Dict[str, Any]], key_fields: List[str]) -> List[List[str]]:
    """Return lists of entity names that share identical parameters."""
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for ent in entities:
        key = tuple((f, json.dumps(ent.get(f), sort_keys=True)) for f in key_fields)
        groups[key].append(ent.get("name", ""))
    return [names for names in groups.values() if len(names) > 1]


def load_config(path: str = "schedule-config.json") -> Dict[str, Any]:
    """Load configuration file and apply defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

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
        if not isinstance(item, list) or len(item) < 5:
            raise ValueError("Lesson entry must contain day, slot, subject, cabinet(s) and teachers")
        day, slot, subject_id, cabinet, teachers = item
        if isinstance(cabinet, str):
            cabinet = [cabinet]
        lessons_parsed.append(
            {
                "day": day,
                "slot": int(slot),
                "subject": subject_id,
                "cabinets": cabinet,
                "teachers": list(teachers),
            }
        )
    data["lessons"] = lessons_parsed

    model_conf = data.get("model", {})
    model_conf.setdefault("maxTime", DEFAULT_MAX_TIME)
    model_conf.setdefault("workers", DEFAULT_WORKERS)
    model_conf.setdefault("showProgress", DEFAULT_SHOW_PROGRESS)
    data["model"] = model_conf

    return data


def _init_schedule(days: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Prepare empty schedule structure."""
    schedule = {}
    for day in days:
        name = day["name"]
        schedule[name] = {slot: [] for slot in day["slots"]}
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

    used_idx: Dict[str, int] = defaultdict(int)
    fixed: Dict[tuple, Dict[str, Any]] = {}
    for entry in lessons:
        day = entry["day"]
        slot = int(entry["slot"])
        sid = entry["subject"]
        rooms = entry["cabinets"]
        if isinstance(rooms, str):
            rooms = [rooms]
        tlist = entry["teachers"]

        if day not in day_lookup:
            raise ValueError(f"Unknown day '{day}' in lesson {entry}")
        if slot not in day_lookup[day]:
            raise ValueError(f"Slot {slot} not available on {day}")
        if sid not in subjects:
            raise ValueError(f"Unknown subject '{sid}' in lesson {entry}")
        subj = subjects[sid]

        idx = used_idx[sid]
        if idx >= len(subj.get("classes", [])):
            raise ValueError(f"Too many fixed lessons for subject {sid}")
        length = subj["classes"][idx]
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
                raise ValueError(
                    f"Subject {sid} not permitted in cabinet '{room}'"
                )
        total_capacity = sum(cabinets[r]["capacity"] for r in rooms)
        if total_capacity < class_size:
            raise ValueError(
                f"Cabinets {rooms} too small for subject {sid} (size {class_size})"
            )

        required = int(subj.get("requiredTeachers", 1))
        if len(tlist) != required:
            raise ValueError(
                f"Subject {sid} requires {required} teachers, got {len(tlist)}"
            )

        for t in tlist:
            if t not in teacher_map or sid not in teacher_map[t]:
                raise ValueError(f"Teacher {t} cannot teach subject {sid}")
            if slot not in teacher_limits[t][day]:
                raise ValueError(
                    f"Teacher {t} not available at {day} slot {slot} for subject {sid}"
                )

        for stu in students_by_subject.get(sid, []):
            if slot not in student_limits[stu][day]:
                raise ValueError(
                    f"Student {stu} not available at {day} slot {slot} for subject {sid}"
                )

        primary = set(subj.get("primaryTeachers", []))
        if primary and not primary.issubset(set(tlist)):
            missing = ", ".join(sorted(primary - set(tlist)))
            raise ValueError(
                f"Lesson for subject {sid} missing primary teacher(s): {missing}"
            )

        fixed[(sid, idx)] = {
            "day": day,
            "day_idx": day_index[day],
            "start": slot,
            "length": length,
            "cabinets": rooms,
            "teachers": tlist,
        }
        used_idx[sid] += 1

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
    # teacher and cabinet selections for every class
    teacher_choice: Dict[tuple, cp_model.BoolVar] = {}
    cabinet_choice: Dict[tuple, cp_model.BoolVar] = {}
    allowed_teacher_map: Dict[tuple, List[str]] = {}
    allowed_cabinet_map: Dict[tuple, List[str]] = {}
    avoid_map = {
        sid: subj.get("avoidConsecutive", default_avoid_consecutive)
        for sid, subj in subjects.items()
    }
    # helper mapping for allowed teacher and cabinet choices
    for sid, subj in subjects.items():
        allowed_teachers = subject_teachers.get(sid)
        if not allowed_teachers:
            raise ValueError(f"No teacher available for subject {sid}")
        enrolled = students_by_subject.get(sid, [])
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
                diff = abs(fixed["start"] - subj.get("optimalSlot", 0))
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
                    for t in fixed["teachers"]
                }
                cand_list.append(
                    {
                        "var": var,
                        "day": fixed["day"],
                        "day_idx": fixed["day_idx"],
                        "start": fixed["start"],
                        "length": length,
                        "size": class_size,
                        "students": enrolled,
                        "student_pen": stud_pen_map,
                        "teacher_pen": teach_pen_map,
                        "penalty": sum(stud_pen_map.values())
                        + sum(teach_pen_map.values()),
                        "available_teachers": fixed["teachers"],
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
                                "students": enrolled,
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
            tv_list = []
            for t in allowed_teachers:
                tv = model.NewBoolVar(f"is_{sid}_{idx}_teacher_{t}")
                if fixed is not None:
                    if t in fixed["teachers"]:
                        model.Add(tv == 1)
                    else:
                        model.Add(tv == 0)
                elif t in primary:
                    model.Add(tv == 1)
                teacher_choice[(sid, idx, t)] = tv
                tv_list.append(tv)
            if tv_list:
                model.Add(sum(tv_list) == required)
            allowed_teacher_map[key] = allowed_teachers
            cv_list = []
            for c in allowed_cabinets:
                cv = model.NewBoolVar(f"is_{sid}_{idx}_cab_{c}")
                if fixed is not None:
                    if c in fixed["cabinets"]:
                        model.Add(cv == 1)
                    else:
                        model.Add(cv == 0)
                cabinet_choice[(sid, idx, c)] = cv
                cv_list.append(cv)
            if not cv_list or len(cv_list) < required_cabs:
                raise RuntimeError(f"No cabinet for subject {sid} class {idx}")
            model.Add(sum(cv_list) == required_cabs)
            model.Add(
                sum(
                    cabinets[c]["capacity"] * cabinet_choice[(sid, idx, c)]
                    for c in allowed_cabinets
                )
                >= class_size
            )
            allowed_cabinet_map[key] = allowed_cabinets
            # create interval objects for candidates
            for cand in cand_list:
                cand["interval"] = model.NewOptionalIntervalVar(
                    cand["start"],
                    cand["length"],
                    cand["start"] + cand["length"],
                    cand["var"],
                    f"int_{sid}_{idx}_{cand['day']}_{cand['start']}"
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
                    c["var"]
                    for c in candidates[(sid, idx)]
                    if c["day"] == day["name"]
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

    for (sid, idx), cand_list in candidates.items():
        for cand in cand_list:
            base_int = cand["interval"]
            start = cand["start"]
            end = cand["start"] + cand["length"]
            for stu in cand["students"]:
                student_intervals[stu][cand["day"]].append(
                    (base_int, start, end, cand["var"])
                )
            for t in allowed_teacher_map[(sid, idx)]:
                if t not in cand.get("available_teachers", []):
                    model.AddImplication(cand["var"], teacher_choice[(sid, idx, t)].Not())
                    continue
                pres = model.NewBoolVar(
                    f"teach_{sid}_{idx}_{t}_{cand['day']}_{cand['start']}"
                )
                model.Add(pres <= cand["var"])
                model.Add(pres <= teacher_choice[(sid, idx, t)])
                model.Add(pres >= cand["var"] + teacher_choice[(sid, idx, t)] - 1)
                interval = model.NewOptionalIntervalVar(
                    start,
                    cand["length"],
                    end,
                    pres,
                    f"tint_{sid}_{idx}_{t}_{cand['day']}_{cand['start']}"
                )
                teacher_intervals[t][cand["day"]].append((interval, start, end, pres))
                cand.setdefault("teacher_pres", {})[t] = pres
            for cab in cabinets:
                if (sid, idx, cab) not in cabinet_choice:
                    continue
                cv = cabinet_choice[(sid, idx, cab)]
                pres = model.NewBoolVar(
                    f"cab_{sid}_{idx}_{cab}_{cand['day']}_{cand['start']}"
                )
                model.Add(pres <= cand["var"])
                model.Add(pres <= cv)
                model.Add(pres >= cand["var"] + cv - 1)
                interval = model.NewOptionalIntervalVar(
                    cand["start"],
                    cand["length"],
                    cand["start"] + cand["length"],
                    pres,
                    f"cint_{sid}_{idx}_{cab}_{cand['day']}_{cand['start']}"
                )
                cabinet_intervals[cab][cand["day"]].append((interval, start, end, pres))

    for t, day_map in teacher_intervals.items():
        for ivs in day_map.values():
            if ivs:
                model.AddNoOverlap([iv[0] for iv in ivs])
    for c, day_map in cabinet_intervals.items():
        for ivs in day_map.values():
            if ivs:
                model.AddNoOverlap([iv[0] for iv in ivs])
    for s, day_map in student_intervals.items():
        for ivs in day_map.values():
            if ivs:
                model.AddNoOverlap([iv[0] for iv in ivs])

    # build slot variables for teachers and students based on intervals
    teacher_slot = {}
    student_slot = {}
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
                        parts = [teacher_slot[(t, dname, slots[idx - j])] for j in range(k)]
                        st = model.NewBoolVar(f"tstreak_{t}_{dname}_{s}_{k}")
                        for p in parts:
                            model.Add(st <= p)
                        model.Add(st >= sum(parts) - k + 1)
                        if t_inc[k - 1] > 0:
                            teacher_streak_exprs[t].append(
                                st * t_inc[k - 1] * teacher_importance[t] * teacher_as_students
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
                        parts = [student_slot[(sname, dname, slots[idx - j])] for j in range(k)]
                        st = model.NewBoolVar(f"sstreak_{sname}_{dname}_{s}_{k}")
                        for p in parts:
                            model.Add(st <= p)
                        model.Add(st >= sum(parts) - k + 1)
                        if s_inc[k - 1] > 0:
                            student_streak_exprs[sname].append(
                                st * s_inc[k - 1] * student_importance[sname] * student_size[sname]
                            )
    teacher_streak_exprs = {k: sum(v) if v else 0 for k, v in teacher_streak_exprs.items()}
    student_streak_exprs = {k: sum(v) if v else 0 for k, v in student_streak_exprs.items()}

    # penalties for consecutive days of the same subject
    consecutive_vars = []
    consecutive_map = {}
    for sid, subj in subjects.items():
        if not avoid_map.get(sid, True):
            continue
        count = len(subj["classes"])
        if count <= 1:
            continue
        for j in range(count):
            pair_vars = []
            for i in range(count):
                if i == j:
                    continue
                var = model.NewBoolVar(f"cons_{sid}_{i}_{j}")
                model.Add(class_day_idx[(sid, j)] == class_day_idx[(sid, i)] + 1).OnlyEnforceIf(var)
                model.Add(class_day_idx[(sid, j)] != class_day_idx[(sid, i)] + 1).OnlyEnforceIf(var.Not())
                pair_vars.append(var)
            if pair_vars:
                any_v = model.NewBoolVar(f"cons_any_{sid}_{j}")
                model.AddMaxEquality(any_v, pair_vars)
                consecutive_vars.append(any_v)
                consecutive_map[(sid, j)] = any_v

    # objective
    model_params = cfg.get("model", {})
    obj_mode = settings.get("objective", ["total"])[0]

    teacher_gap_exprs = {
        t: gap_teacher_val * teacher_importance[t]
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
            tv = teacher_choice.get((sid, j, t))
            if tv is None:
                continue
            both = model.NewBoolVar(f"cons_t_{sid}_{j}_{t}")
            model.AddMultiplicationEquality(both, [var, tv])
            expr.append(
                both
                * consecutive_pen_val
                * teacher_importance[t]
                * teacher_as_students
            )
        teacher_consec_exprs[t] = sum(expr) if expr else 0

    consecutive_expr = sum(teacher_consec_exprs.values()) if consecutive_vars else 0

    if obj_mode == "total":
        total_expr = (
            sum(teacher_gap_exprs.values())
            + sum(teacher_unopt_exprs.values())
            + sum(teacher_consec_exprs.values())
            + sum(teacher_streak_exprs.values())
            + sum(student_gap_exprs.values())
            + sum(student_unopt_exprs.values())
            + sum(student_streak_exprs.values())
        )
        model.Minimize(total_expr)
    else:
        approx_bound = (
            sum(c["penalty"] for cl in candidates.values() for c in cl)
            + len(teacher_gap_vars)
            * gap_teacher_val
            * max(teacher_importance.values() or [0])
            + len(student_gap_vars)
            * gap_student_val
            * max(student_importance.values() or [0])
            * max(student_size.values() or [1])
            + len(consecutive_vars)
            * consecutive_pen_val
            * teacher_as_students
            * max(teacher_importance.values() or [0])
            + len(teacher_slot)
            * (max(teacher_streak_list or [0]))
            * teacher_as_students
            * max(teacher_importance.values() or [0])
            + len(student_slot)
            * (max(student_streak_list or [0]))
            * max(student_importance.values() or [0])
            * max(student_size.values() or [1])
        )
        max_pen = model.NewIntVar(0, int(approx_bound + 1), "maxPenalty")
        for t in teacher_importance:
            model.Add(
                teacher_gap_exprs[t]
                + teacher_unopt_exprs[t]
                + teacher_consec_exprs[t]
                + teacher_streak_exprs[t]
                <= max_pen
            )
        for s in student_importance:
            model.Add(
                student_gap_exprs[s]
                + student_unopt_exprs[s]
                + student_streak_exprs[s]
                <= max_pen
            )
        model.Minimize(max_pen)

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
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")

    schedule = _init_schedule(days)
    for (sid, idx), cand_list in candidates.items():
        selected_cabs = [
            c
            for c in allowed_cabinet_map[(sid, idx)]
            if solver.Value(cabinet_choice[(sid, idx, c)])
        ]
        assigned_teachers = [
            t
            for t in allowed_teacher_map[(sid, idx)]
            if solver.Value(teacher_choice[(sid, idx, t)])
        ]
        for c in cand_list:
            if solver.Value(c["var"]):
                for s in range(c["start"], c["start"] + c["length"]):
                    schedule[c["day"]][s].append(
                        {
                            "subject": sid,
                            "teachers": assigned_teachers,
                            "cabinets": selected_cabs,
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
    teacher_as_students = settings.get("teacherAsStudents", [15])[0]

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
                for stu in cls["students"]:
                    student_slots[stu][name].add(slot)

    teacher_state = {
        t: {day["name"]: {} for day in cfg["days"]} for t in teacher_names
    }
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

    student_state = {
        s: {day["name"]: {} for day in cfg["days"]} for s in student_names
    }
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

    penalties_cfg = {k: v[0] for k, v in cfg.get("penalties", {}).items()}
    def_teacher_imp = defaults.get("teacherImportance", [1])[0]
    def_student_imp = defaults.get("studentImportance", [0])[0]
    teacher_importance = {
        t["name"]: t.get("importance", def_teacher_imp)
        for t in cfg.get("teachers", [])
    }
    student_importance = {
        s["name"]: s.get("importance", def_student_imp)
        for s in cfg.get("students", [])
    }
    default_opt = defaults.get("optimalSlot", [0])[0]

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
        teach_count = {t: 0 for t in teacher_names}
        stud_count = {s: 0 for s in student_names}
        for slot in day["slots"]:
            # gap penalties for teachers
            for t in teacher_names:
                if teacher_state[t][dname][slot] == "gap":
                    p = penalties_cfg.get("gapTeacher", 0) * teacher_importance[t]
                    slot_penalties[dname][slot]["gapTeacher"] += p
                    slot_penalty_details[dname][slot].append({"name": t, "type": "gapTeacher", "amount": p})
                if teacher_state[t][dname][slot] == "class":
                    teach_count[t] += 1
                    val_idx = min(len(teacher_streak_list) - 1, teach_count[t] - 1) if teacher_streak_list else -1
                    if val_idx >= 0:
                        pen = teacher_streak_list[val_idx]
                        if pen:
                            amount = pen * teacher_importance[t] * teacher_as_students
                            slot_penalties[dname][slot]["teacherLessonStreak"] += amount
                            slot_penalty_details[dname][slot].append({"name": t, "type": "teacherLessonStreak", "amount": amount})
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
                    slot_penalty_details[dname][slot].append({"name": sname, "type": "gapStudent", "amount": p})
                if student_state[sname][dname][slot] == "class":
                    stud_count[sname] += 1
                    val_idx = min(len(student_streak_list) - 1, stud_count[sname] - 1) if student_streak_list else -1
                    if val_idx >= 0:
                        pen = student_streak_list[val_idx]
                        if pen:
                            amount = pen * student_importance[sname] * student_size.get(sname, 1)
                            slot_penalties[dname][slot]["studentLessonStreak"] += amount
                            slot_penalty_details[dname][slot].append({"name": sname, "type": "studentLessonStreak", "amount": amount})
                else:
                    stud_count[sname] = 0

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
                    slot_penalty_details[dname][slot].append({"name": t, "type": "unoptimalSlot", "amount": p})
                for sname in cls["students"]:
                    p = (
                        base
                        * student_importance[sname]
                        * student_size.get(sname, 1)
                    )
                    slot_penalties[dname][slot]["unoptimalSlot"] += p
                    slot_penalty_details[dname][slot].append({"name": sname, "type": "unoptimalSlot", "amount": p})

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
                        slot_penalty_details[dname][slot].append({"name": t, "type": "consecutiveClass", "amount": p})

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
                sid = cls["subject"]
                count = cls.get("size", len(cls.get("students", [])))
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
                for tid in cls.get("teachers", []):
                    subjects[sid]["teachers"].add(tid)
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

    teacher_names = {
        t["name"]: t.get("printName", t.get("name", t["name"]))
        for t in cfg.get("teachers", [])
    }
    student_names = {s["name"]: s.get("name", s["name"]) for s in cfg.get("students", [])}
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

            header = f"  Lesson {idx+1} [gap T:{gap_t} S:{gap_s} home T:{home_t} S:{home_s}]"
            if not classes:
                print(f"{header}: --")
                continue
            print(f"{header}:")
            for cls in classes:
                subj = subject_names.get(cls["subject"], cls["subject"])
                teacher = ", ".join(teacher_names.get(t, t) for t in cls.get("teachers", []))
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
<title>Schedule Overview</title>
<style>
body { font-family: Arial, sans-serif; }
.schedule-grid { border-collapse: collapse; width: 100%; display:grid; }
.schedule-grid .cell, .schedule-grid .header { border:1px solid #999; padding:4px; vertical-align:top; }
.schedule-grid .cell { display:flex; flex-direction:column; }
.schedule-grid .header { background:#f0f0f0; text-align:center; }
.mini-grid { margin-top:10px; }
.class-block { display:flex; flex-direction:column; margin-bottom:4px; }
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
.person-hours{flex:0 0 6%;text-align:center;}
.person-time{flex:0 0 6%;text-align:center;}
.person-subjects{flex:0 0 51%;text-align:left;padding:0;}
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
</style>
</head>
<body>
<h1>Schedule Overview</h1>
<div id="table" class="schedule-grid"></div>
<h2 class="overview-section">Teachers Overview</h2>
<div id="teachers" class="overview-table"></div>
<h2 class="overview-section">Students Overview</h2>
<div id="students" class="overview-table"></div>
<div id="modal" class="modal"><div class="modal-content"><div class="modal-header"><div id="history" class="history"></div><span id="close" class="close">&#10006;</span></div><div id="modal-body"></div></div></div>
<script>
const scheduleData = __SCHEDULE__;
const configData = __CONFIG__;
</script>
<script>
(function(){
const modal=document.getElementById('modal');
const close=document.getElementById('close');
const historyBox=document.getElementById('history');
const modalBody=document.getElementById('modal-body');
close.onclick=()=>{modal.style.display='none';};
window.onclick=e=>{if(e.target==modal)modal.style.display='none';};
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
function teacherSpan(name,subj){
  const id=teacherIndex[name];
  const prim=(configData.subjects[subj]||{}).primaryTeachers||[];
  const disp=teacherDisplay[name]||name;
  const inner=prim.includes(name)?'<strong>'+disp+'</strong>':disp;
  return '<span class="clickable teacher" data-id="'+id+'">'+inner+'</span>';
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
   const hdr=document.createElement('div');hdr.className='header';hdr.textContent='Lesson '+(i+1);container.appendChild(hdr);
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i) || {classes:[],gaps:{students:[],teachers:[]},home:{students:[],teachers:[]},penalty:{}};
     const cell=document.createElement('div');
     cell.className='cell';
     const pVal=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
     minP=Math.min(minP,pVal);maxP=Math.max(maxP,pVal);
     slot.classes.forEach(cls=>{
       const block=document.createElement('div');
       block.className='class-block';
       const subj=subjectDisplay[cls.subject]||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const l1=document.createElement('div');
       l1.className='class-line';
        const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
        l1.innerHTML='<span class="cls-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
         '<span class="cls-room">'+rooms+'</span>'+
         '<span class="cls-part">'+part+'</span>';
       const l2=document.createElement('div');
       l2.className='class-line';
      const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ');
       l2.innerHTML='<span class="cls-teach">'+tNames+'</span>'+
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

function makeGrid(filterFn){
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 let html='<div class="schedule-grid mini-grid" style="grid-template-columns:auto repeat('+scheduleData.days.length+',1fr)">';
 html+='<div></div>';
 scheduleData.days.forEach(d=>{html+='<div class="header">'+d.name+'</div>';});
 for(let i=0;i<maxSlots;i++){
  html+='<div class="header">Lesson '+(i+1)+'</div>';
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i)||{classes:[]};
     html+='<div class="cell">';
     slot.classes.filter(filterFn).forEach(cls=>{
       const subj=subjectDisplay[cls.subject]||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ');
        const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
        html+='<div class="class-block">'+
         '<div class="class-line">'+
          '<span class="cls-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
          '<span class="cls-room">'+rooms+'</span>'+
          '<span class="cls-part">'+part+'</span>'+
        '</div>'+
        '<div class="class-line">'+
          '<span class="cls-teach">'+tNames+'</span>'+
          '<span class="cls-size">'+cls.size+'</span>'+
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
 slot.classes.forEach((cls)=>{
   const subj=subjectDisplay[cls.subject]||cls.subject;
   const part=(cls.length>1)?((idx-cls.start+1)+'/'+cls.length):'1/1';
   html+='<div class="slot-class">'+
     '<div class="detail-line">'+
       '<span class="detail-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
      '<span class="detail-teacher">'+(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ')+'</span>'+
       '<span class="detail-room">'+(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ')+'</span>'+
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
        const isTeach=p.type==='gapTeacher'||(p.type==='unoptimalSlot'&&teacherSet.has(p.name))||(p.type==='consecutiveClass'&&teacherSet.has(p.name));
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
     teachSlots.forEach(sl=>{const c=sl.classes.find(x=>(x.teachers||[]).includes(name));sizes.push(c.size);total++;});
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
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let hours=0,gap=0,time=0,subjects={},pen=0;
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
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
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
   const pr=item.info.importance!==undefined?item.info.importance:(configData.defaults.teacherImportance||[1])[0];
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
    const name=subjectDisplay[sid]||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+s.avg+'</span></div>';
   });
   row.innerHTML='<span class="person-name clickable teacher" data-id="'+teacherIndex[item.info.name]+'">'+(teacherDisplay[item.info.name]||item.info.name)+'</span>'+
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
   const pr=item.info.importance!==undefined?item.info.importance:(configData.defaults.studentImportance||[0])[0];
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
 const display=teacherDisplay[name]||name;
 const defImp=(configData.defaults.teacherImportance||[1])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const boldImp=imp!==defImp;
 const stats=computeTeacherStats(name);
 const full=computeTeacherInfo(name);
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const boldArr=full.arrive!==defArr;
 let html='<h2>Teacher: '+display+'</h2>'+makeGrid(cls=>cls.teachers.includes(name));
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
 ['Total classes',stats.totalClasses],
 ['Average size',stats.avgSize],
 ['Penalty',full.penalty.toFixed(1)]
 ];
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
let html='<h2>Student: '+name+'</h2>'+makeGrid(cls=>cls.students.includes(name));
 html+='<h3>Subjects</h3><table class="info-table"><tr><th>Subject</th><th>Classes</th><th>Penalty</th></tr>';
Object.keys(full.subjects).forEach(sid=>{const s=full.subjects[sid];const sn=subjectDisplay[sid]||sid;html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></td><td class="num">'+s.count+'</td><td class="num">'+(s.penalty||0).toFixed(1)+'</td></tr>';});
 html+='</table>';
 const params=[
 ['Group size',group,boldGroup],
 ['Priority',imp,boldImp],
 ['Arrive early',full.arrive?'yes':'no',boldArr],
 ['Gap hours',stats.gap],
 ['At school',stats.time],
 ['Penalty',full.penalty.toFixed(1)]
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showCabinet(name,fromModal=false){
 const info=configData.cabinets[name]||{};
  const disp=cabinetDisplay[name]||name;
  let html='<h2>Room: '+disp+'</h2>'+makeGrid(cls=>(cls.cabinets||[]).includes(name));
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
 let html='<h2>Subject: '+disp+'</h2>'+makeGrid(cls=>cls.subject===id);
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
    html = html.replace("__SCHEDULE__", schedule_json).replace("__CONFIG__", cfg_json)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-y"]
    auto_yes = "-y" in sys.argv[1:]

    cfg_path = args[0] if len(args) >= 1 else "schedule-config.json"
    out_path = args[1] if len(args) >= 2 else "schedule.json"
    html_path = args[2] if len(args) >= 3 else "schedule.html"

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

    student_dups = _detect_duplicates(
        cfg.get("students", []), ["subjects"]
    )
    if student_dups and not skip_solve:
        print("Duplicate entities detected:")
        for names in student_dups:
            print(f"Students: {', '.join(names)}")
        if not auto_yes:
            ans = input("Continue despite duplicates? [y/N] ")
            if not ans.strip().lower().startswith("y"):
                print("Exiting due to duplicates.")
                return
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
        ans = input("Show analysis and generate html page? [y/N] ")
        show_analysis = ans.strip().lower().startswith("y")

    if show_analysis:
        report_analysis(result, cfg)
        generate_html(result, cfg, html_path)
        print(f"{html_path} generated")


if __name__ == "__main__":
    main()
