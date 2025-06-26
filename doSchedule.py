#!/usr/bin/env python3
"""Combined schedule utility.
Generates mock data, solves schedule and optionally analyses results.
"""

import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean

from faker import Faker
from ortools.sat.python import cp_model
import math

# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def generate_mock_data(extra_teacher_prob: float = 0.1):
    fake = Faker()
    data = {
        "limits": {
            "MAX_CLASSES_PER_BLOCK": 10,
            "MAX_STUDENTS_PER_CLASS": 15,
            "MIN_STUDENTS_PER_CLASS": 1,
            "MAX_BLOCKS": 40,
        },
        "teachers": {},
        "subjects": {},
        "students": {},
    }

    senior_teachers = [f"teacher_s{i}" for i in range(1, 11)]
    for tid in senior_teachers:
        data["teachers"][tid] = {"name": fake.name()}

    years = ["DP1", "DP2"]
    subject_types = {
        "simple": {"hours": 3, "count": 6},
        "advanced": {"hours": 5, "count": 6},
        "extra": {"hours": 3, "count": 1},
    }

    for year in years:
        for stype, cfg in subject_types.items():
            for i in range(1, cfg["count"] + 1):
                sid = f"{stype}_{year}_{i}"
                main_teacher = random.choice(senior_teachers)
                teachers_list = [main_teacher]
                if random.random() < extra_teacher_prob and stype != "extra":
                    other = [t for t in senior_teachers if t != main_teacher]
                    if other:
                        teachers_list.append(random.choice(other))
                data["subjects"][sid] = {
                    "name": f"{stype.capitalize()} {year} Subject {i}",
                    "hours": cfg["hours"],
                    "teachers": teachers_list,
                }

    student_count = 0
    for year in years:
        for i in range(1, 16):
            uid = f"student_{year}_{i}"
            student_count += 1
            simple_subjects = [f"simple_{year}_{j}" for j in range(1, 7)]
            advanced_subjects = [f"advanced_{year}_{j}" for j in range(1, 7)]
            selected_simple = random.sample(simple_subjects, 3)
            selected_advanced = random.sample(advanced_subjects, 3)
            extra_subject = [f"extra_{year}_1"]
            data["students"][uid] = {
                "name": fake.name(),
                "subjects": selected_simple + selected_advanced + extra_subject,
            }

    print(
        f"Generated mock data with {len(data['teachers'])} teachers, "
        f"{len(data['subjects'])} subjects and {student_count} students",
    )
    return data

# ---------------------------------------------------------------------------
# Solver utilities
# ---------------------------------------------------------------------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    limits = data["limits"]
    limits.setdefault("MAX_BLOCKS", 40)
    limits.setdefault("MIN_STUDENTS_PER_CLASS", 1)
    return data, limits


def upper_bound_blocks(data, limits):
    total_hours = 0
    for sid, subj in data["subjects"].items():
        h = subj["hours"]
        enrolled = sum(sid in stu["subjects"] for stu in data["students"].values())
        total_hours += h * enrolled
    return min(
        limits["MAX_BLOCKS"],
        math.ceil(total_hours / limits["MAX_CLASSES_PER_BLOCK"]),
    )


def build_model(data, limits):
    model = cp_model.CpModel()
    subjects = list(data["subjects"])
    students = list(data["students"])
    teachers_all = list(data["teachers"])

    teach_map = {}
    for s in subjects:
        subj = data["subjects"][s]
        if "teachers" in subj:
            teach_map[s] = subj["teachers"]
        elif "teacher" in subj:
            teach_map[s] = [subj["teacher"]]
        else:
            raise KeyError(f"Subject '{s}' has no 'teacher' or 'teachers' field")

    pairs = [(s, t) for s in subjects for t in teach_map[s]]
    B = upper_bound_blocks(data, limits)

    y = {}
    for b in range(B):
        for s, t in pairs:
            for u in students:
                if s in data["students"][u]["subjects"]:
                    y[(b, s, t, u)] = model.NewBoolVar(f"y_b{b}_{s}_{t}_{u}")

    class_used = {
        (b, s, t): model.NewBoolVar(f"class_b{b}_{s}_{t}")
        for b in range(B)
        for s, t in pairs
    }
    block_used = {b: model.NewBoolVar(f"block_{b}") for b in range(B)}

    for b in range(B):
        for u in students:
            model.Add(sum(y.get((b, s, t, u), 0) for s, t in pairs) <= 1)

    for u in students:
        subj_list = data["students"][u]["subjects"]
        for s in subj_list:
            need = data["subjects"][s]["hours"]
            model.Add(
                sum(y[(b, s, t, u)] for b in range(B) for t in teach_map[s] if (b, s, t, u) in y)
                == need
            )

    max_sz = limits["MAX_STUDENTS_PER_CLASS"]
    min_sz = limits["MIN_STUDENTS_PER_CLASS"]
    for b in range(B):
        for s, t in pairs:
            enrol = [y[(b, s, t, u)] for u in students if (b, s, t, u) in y]
            if enrol:
                total = sum(enrol)
                model.Add(total >= min_sz).OnlyEnforceIf(class_used[(b, s, t)])
                model.Add(total <= max_sz * class_used[(b, s, t)])
                model.Add(total >= 1).OnlyEnforceIf(class_used[(b, s, t)])
            else:
                model.Add(class_used[(b, s, t)] == 0)

    for b in range(B):
        for t in teachers_all:
            model.Add(sum(class_used[(b, s, t)] for s in subjects if (s, t) in pairs) <= 1)

    for b in range(B):
        model.Add(sum(class_used[(b, s, t)] for s, t in pairs) <= limits["MAX_CLASSES_PER_BLOCK"])
        model.AddMaxEquality(block_used[b], [class_used[(b, s, t)] for s, t in pairs])

    model.Add(sum(block_used[b] for b in range(B)) <= limits["MAX_BLOCKS"])
    model.Minimize(sum(block_used[b] for b in range(B)))

    return model, y, class_used, block_used, B, pairs, students


def solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    solver.parameters.num_search_workers = 10
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible schedule found")
    return solver


def extract(solver, y, B, pairs, students):
    schedule = []
    for b in range(B):
        block = []
        for s, t in pairs:
            stu_list = [u for u in students if (b, s, t, u) in y and solver.Value(y[(b, s, t, u)]) == 1]
            if stu_list:
                block.append({"subject": s, "teacher": t, "students": stu_list})
        if block:
            schedule.append(block)
    return schedule

# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def analyse_teachers(schedule):
    teachers = defaultdict(lambda: {"blocks": 0, "students": [], "subjects": defaultdict(list)})
    for block in schedule:
        for cls in block:
            t = cls["teacher"]
            s = cls["subject"]
            n = len(cls["students"])
            teachers[t]["blocks"] += 1
            teachers[t]["students"].append(n)
            teachers[t]["subjects"][s].append(n)
    return teachers


def analyse_students(schedule):
    student_hours = defaultdict(lambda: defaultdict(int))
    for block in schedule:
        for cls in block:
            subj = cls["subject"]
            for stu in cls["students"]:
                student_hours[stu][subj] += 1
    return student_hours


def analyse_subjects(schedule):
    subjects = defaultdict(lambda: {"students": set(), "teachers": set(), "class_sizes": []})
    for block in schedule:
        for cls in block:
            subj = cls["subject"]
            subjects[subj]["students"].update(cls["students"])
            subjects[subj]["teachers"].add(cls["teacher"])
            subjects[subj]["class_sizes"].append(len(cls["students"]))
    return subjects


def _split_two_parts(name: str, max_len: int = 12) -> tuple[str, str]:
    """Split a long header name into two roughly equal parts."""
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
    """Return string representing a simple ascii table with optional two-line header.

    When ``join_rows`` is True the ``rows`` argument is interpreted as pairs of
    rows that represent logically single table rows split in two lines. In that
    case horizontal separators are printed only after each pair, producing a
    visual effect of cells spanning two lines.
    """
    col_count = max(len(r) for r in rows)
    if header_top:
        col_count = max(col_count, len(header_top))
    if header_bottom:
        col_count = max(col_count, len(header_bottom))

    # normalize rows
    rows = [list(r) + [""] * (col_count - len(r)) for r in rows]
    if header_top:
        header_top = list(header_top) + [""] * (col_count - len(header_top))
    if header_bottom:
        header_bottom = list(header_bottom) + [""] * (col_count - len(header_bottom))

    if center_mask is None:
        center_mask = [False] * col_count
    else:
        center_mask = list(center_mask) + [False] * (col_count - len(center_mask))

    # compute widths
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
    for idx, r in enumerate(rows):
        lines.append(fmt_row(r))
        if join_rows and idx % 2 == 0 and idx + 1 < len(rows):
            continue
        lines.append(sep)
    return "\n".join(lines)


def _teacher_table(teachers, teacher_names, subject_names):
    rows = []
    max_subj = max((len(info["subjects"]) for info in teachers.values()), default=0)
    for tid, info in sorted(teachers.items(), key=lambda x: x[1]["blocks"], reverse=True):
        name1, name2 = _split_two_parts(teacher_names.get(tid, tid))
        row_top = [name1]
        row_bottom = [name2]
        subj_stats = sorted(info["subjects"].items(), key=lambda x: len(x[1]), reverse=True)
        for sid, counts in subj_stats:
            row_top.append(subject_names.get(sid, sid))
            avg_s = mean(counts) if counts else 0
            row_bottom.append(f"{len(counts)} | {avg_s:.1f}")
        # pad to max subjects
        row_top += [""] * (max_subj - len(subj_stats))
        row_bottom += [""] * (max_subj - len(subj_stats))
        rows.append(row_top)
        rows.append(row_bottom)

    header_top = ["Teacher"] + [f"Subject #{i+1}" for i in range(max_subj)]
    header_bottom = [""] + ["Classes | Avg" for _ in range(max_subj)]

    center_mask = [False] + [True] * max_subj

    return _format_table(rows, header_top, header_bottom, center_mask, join_rows=True)


def _student_table(students, student_names, subject_names):
    """Return formatted table of student schedules."""
    max_subj = max((len(info) for info in students.values()), default=0)

    header_top = ["Student"] + [f"Subject #{i+1}" for i in range(max_subj)] + ["Total"]
    header_bottom = ["Name"] + ["Lessons" for _ in range(max_subj)] + ["Hours"]

    rows = []
    for sid in sorted(students.keys(), key=lambda x: student_names.get(x, x)):
        name1, name2 = _split_two_parts(student_names.get(sid, sid))
        row_top = [name1]
        row_bottom = [name2]
        subj_map = students[sid]
        total_hours = 0
        for sub in sorted(subj_map.keys(), key=lambda x: subject_names.get(x, x)):
            row_top.append(subject_names.get(sub, sub))
            hours = subj_map[sub]
            row_bottom.append(str(hours))
            total_hours += hours
        # pad to max subjects
        row_top += ["" for _ in range(max_subj - len(subj_map))]
        row_bottom += ["" for _ in range(max_subj - len(subj_map))]
        row_top.append("")
        row_bottom.append(str(total_hours))
        rows.append(row_top)
        rows.append(row_bottom)

    center_mask = [False] + [True] * max_subj + [True]
    return _format_table(rows, header_top, header_bottom, center_mask, join_rows=True)


def _subject_table(subjects, subject_names):
    rows = []
    for sid, info in sorted(subjects.items(), key=lambda x: subject_names.get(x[0], x[0])):
        name = subject_names.get(sid, sid)
        total_students = len(info["students"])
        teachers_cnt = len(info["teachers"])
        avg_size = mean(info["class_sizes"]) if info["class_sizes"] else 0
        rows.append([name, str(total_students), str(teachers_cnt), f"{avg_size:.1f}"])

    header = ["Subject", "Students", "Teachers", "Avg class"]
    center_mask = [False, True, True, True]
    return _format_table(rows, header, center_mask=center_mask)


def report_analysis(schedule, data):
    teachers = analyse_teachers(schedule)
    students = analyse_students(schedule)
    subjects = analyse_subjects(schedule)

    teacher_names = {tid: info.get("name", tid) for tid, info in data.get("teachers", {}).items()}
    student_names = {sid: info.get("name", sid) for sid, info in data.get("students", {}).items()}
    subject_names = {sid: info.get("name", sid) for sid, info in data.get("subjects", {}).items()}

    print("=== Teachers ===")
    print(_teacher_table(teachers, teacher_names, subject_names))

    print("\n=== Students ===")
    print(_student_table(students, student_names, subject_names))

    print("\n=== Subjects ===")
    print(_subject_table(subjects, subject_names))

# ---------------------------------------------------------------------------
# Main combined entry
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if a != "-y"]
    auto_yes = "-y" in sys.argv[1:]

    cfg_path = args[0] if len(args) >= 1 else "config.json"
    out_path = args[1] if len(args) >= 2 else "blocks.js"

    if not os.path.exists(cfg_path):
        if auto_yes:
            do_mock = True
        else:
            ans = input(f"Config '{cfg_path}' not found. Generate mock data? [y/N] ")
            do_mock = ans.strip().lower().startswith("y")
        if not do_mock:
            print("Aborting, no configuration available.")
            return
        data = generate_mock_data()
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Mock data written to {cfg_path}")

    data, limits = load_config(cfg_path)
    model, y, class_used, block_used, B, pairs, students = build_model(data, limits)
    solver = solve(model)
    schedule = extract(solver, y, B, pairs, students)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    print(f"Schedule written to {out_path}")

    if auto_yes:
        show_analysis = True
    else:
        ans = input("Show analysis? [y/N] ")
        show_analysis = ans.strip().lower().startswith("y")
    if show_analysis:
        report_analysis(schedule, data)


if __name__ == "__main__":
    main()
