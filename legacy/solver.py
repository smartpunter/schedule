#!/usr/bin/env python3
"""Schedule generator using Google OR-Tools.

The script reads a configuration JSON describing available teachers,
subjects and students. It builds a constraint model to allocate classes
to time blocks and writes the resulting schedule to ``schedule.json``.

Usage::

    python solver.py <config.json>

Dependencies::

    pip install ortools
"""

import json, sys, math
from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_config(path: str):
    with open(path, "r", encoding="utf‑8") as f:
        data = json.load(f)

    limits = data["limits"]
    limits.setdefault("MAX_BLOCKS", 40)
    limits.setdefault("MIN_STUDENTS_PER_CLASS", 1)
    return data, limits


def upper_bound_blocks(data, limits):
    total_hours = 0
    for subj_id, subj in data["subjects"].items():
        h = subj["hours"]
        enrolled = sum(subj_id in stu["subjects"] for stu in data["students"].values())
        total_hours += h * enrolled
    return min(
        limits["MAX_BLOCKS"],
        math.ceil(total_hours / limits["MAX_CLASSES_PER_BLOCK"]),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(data, limits):
    model = cp_model.CpModel()

    subjects = list(data["subjects"])
    students = list(data["students"])
    teachers_all = list(data["teachers"])

    # map each subject to its available teachers
    teach_map = {}
    for s in subjects:
        subj = data["subjects"][s]
        if "teachers" in subj:
            teach_map[s] = subj["teachers"]
        elif "teacher" in subj:
            teach_map[s] = [subj["teacher"]]  # normalize single teacher field
        else:
            raise KeyError(f"Subject '{s}' has no 'teacher' or 'teachers' field")

    pairs = [(s, t) for s in subjects for t in teach_map[s]]

    B = upper_bound_blocks(data, limits)

    # --- Decision variables -------------------------------------------------
    # y[b,s,t,u] = 1 if student u attends subject s with teacher t in block b
    y = {}
    for b in range(B):
        for s, t in pairs:
            for u in students:
                if s in data["students"][u]["subjects"]:
                    y[(b, s, t, u)] = model.NewBoolVar(f"y_b{b}_{s}_{t}_{u}")

    # class_used[b,s,t]
    class_used = {
        (b, s, t): model.NewBoolVar(f"class_b{b}_{s}_{t}")
        for b in range(B)
        for s, t in pairs
    }

    # block_used[b] indicates whether block b has at least one class
    block_used = {b: model.NewBoolVar(f"block_{b}") for b in range(B)}

    # --- Constraints --------------------------------------------------------
    # 1) A student can attend at most one class per block
    for b in range(B):
        for u in students:
            model.Add(
                sum(
                    y.get((b, s, t, u), 0)
                    for s, t in pairs
                )
                <= 1
            )

    # 2) Required hours per subject
    for u in students:
        subj_list = data["students"][u]["subjects"]
        for s in subj_list:
            need = data["subjects"][s]["hours"]
            model.Add(
                sum(
                    y[(b, s, t, u)]
                    for b in range(B)
                    for t in teach_map[s]
                    if (b, s, t, u) in y
                )
                == need
            )

    # 3) Class capacity and link to class_used
    max_sz = limits["MAX_STUDENTS_PER_CLASS"]
    min_sz = limits["MIN_STUDENTS_PER_CLASS"]

    for b in range(B):
        for s, t in pairs:
            enrol = [
                y[(b, s, t, u)]
                for u in students
                if (b, s, t, u) in y
            ]
            if enrol:  # class may be empty if no one enrolled
                total = sum(enrol)
                model.Add(total >= min_sz).OnlyEnforceIf(class_used[(b, s, t)])
                model.Add(total <= max_sz * class_used[(b, s, t)])
                # if class_used=0 -> total=0
                model.Add(total >= 1).OnlyEnforceIf(class_used[(b, s, t)])
            else:
                # no students -> do not use this class
                model.Add(class_used[(b, s, t)] == 0)

    # 4) A teacher can teach at most one class per block
    for b in range(B):
        for t in teachers_all:
            model.Add(
                sum(
                    class_used[(b, s, t)]
                    for s in subjects
                    if (s, t) in pairs
                )
                <= 1
            )

    # 5) Room capacity per block
    for b in range(B):
        model.Add(
            sum(class_used[(b, s, t)] for s, t in pairs)
            <= limits["MAX_CLASSES_PER_BLOCK"]
        )
        # link block_used with classes present
        model.AddMaxEquality(
            block_used[b],
            [class_used[(b, s, t)] for s, t in pairs]
        )


    # 6) MAX_BLOCKS
    model.Add(sum(block_used[b] for b in range(B)) <= limits["MAX_BLOCKS"])

    # --- Objective: minimise blocks first, then number of classes -------------
    num_blocks = sum(block_used[b] for b in range(B))
    num_classes = sum(class_used[(b, s, t)] for b in range(B) for s, t in pairs)
    big_M = B * limits["MAX_CLASSES_PER_BLOCK"] + 1
    model.Minimize(num_blocks * big_M + num_classes)

    return model, y, class_used, block_used, B, pairs, students


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300  # five minute limit
    solver.parameters.num_search_workers = 10    # use up to ten CPUs
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
            stu_list = [
                u for u in students
                if (b, s, t, u) in y and solver.Value(y[(b, s, t, u)]) == 1
            ]
            if stu_list:
                block.append({
                    "subject": s,
                    "teacher": t,
                    "students": stu_list
                })
        if block:
            schedule.append(block)
    return schedule


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("usage: python solver.py <config.json>", file=sys.stderr)
        sys.exit(1)

    data, limits = load_config(sys.argv[1])
    model, y, class_used, block_used, B, pairs, students = build_model(data, limits)
    solver = solve(model)
    result = extract(solver, y, B, pairs, students)
    with open("schedule.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Schedule written to schedule.json")


if __name__ == "__main__":
    main()
