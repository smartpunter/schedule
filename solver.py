#!/usr/bin/env python3
"""
schedule_multi.py ― оптимальный генератор блоков
------------------------------------------------
* Поддерживает **несколько учителей на предмет** (поле `teachers: [...]`).
* Разрешает **несколько параллельных уроков одного предмета в одном блоке**
  (каждый – с своим учителем).
* Выводит **массив блоков**, каждый блок — массив объектов вида  
  `{ "subject": <subject_id>, "teacher": <teacher_id>, "students": [...] }`.

Требования:
    pip install ortools
Запуск:
    python schedule_multi.py config.json > schedule.json
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

    # для каждого предмета список учителей
    teach_map = {}
    for s in subjects:
        subj = data["subjects"][s]
        if "teachers" in subj:
            teach_map[s] = subj["teachers"]
        elif "teacher" in subj:
            teach_map[s] = [subj["teacher"]]  # Преобразуем строку в список
        else:
            raise KeyError(f"Subject '{s}' has no 'teacher' or 'teachers' field")

    pairs = [(s, t) for s in subjects for t in teach_map[s]]

    B = upper_bound_blocks(data, limits)

    # --- Decision vars ------------------------------------------------------
    # y[b,s,t,u] = 1 если ученик u на уроке (s,t) в блоке b
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

    # block_used[b]
    block_used = {b: model.NewBoolVar(f"block_{b}") for b in range(B)}

    # --- Constraints --------------------------------------------------------
    # 1) Ученик – не более 1 урока в блоке
    for b in range(B):
        for u in students:
            model.Add(
                sum(
                    y.get((b, s, t, u), 0)
                    for s, t in pairs
                )
                <= 1
            )

    # 2) Часы по предмету = hours
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

    # 3) Вместимость классов и связь с class_used
    max_sz = limits["MAX_STUDENTS_PER_CLASS"]
    min_sz = limits["MIN_STUDENTS_PER_CLASS"]

    for b in range(B):
        for s, t in pairs:
            enrol = [
                y[(b, s, t, u)]
                for u in students
                if (b, s, t, u) in y
            ]
            if enrol:  # предмет может быть не выбран никем
                total = sum(enrol)
                model.Add(total >= min_sz).OnlyEnforceIf(class_used[(b, s, t)])
                model.Add(total <= max_sz * class_used[(b, s, t)])
                # если class_used=0 -> total=0
                model.Add(total >= 1).OnlyEnforceIf(class_used[(b, s, t)])
            else:
                # никто не записан – класс использовать бессмысленно
                model.Add(class_used[(b, s, t)] == 0)

    # 4) Учитель ≤1 урока за блок
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

    # 5) Аудитории
    for b in range(B):
        model.Add(
            sum(class_used[(b, s, t)] for s, t in pairs)
            <= limits["MAX_CLASSES_PER_BLOCK"]
        )
        # связь block_used
        model.AddMaxEquality(
            block_used[b],
            [class_used[(b, s, t)] for s, t in pairs]
        )


    # 6) MAX_BLOCKS
    model.Add(sum(block_used[b] for b in range(B)) <= limits["MAX_BLOCKS"])

    # --- Objective: minimise #used blocks -----------------------------------
    model.Minimize(sum(block_used[b] for b in range(B)))

    return model, y, class_used, block_used, B, pairs, students


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0           # без лимита
    solver.parameters.num_search_workers = 8            # autodetect (≥2)
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
        print("usage: python schedule_multi.py <config.json>", file=sys.stderr)
        sys.exit(1)

    data, limits = load_config(sys.argv[1])
    model, y, class_used, block_used, B, pairs, students = build_model(data, limits)
    solver = solve(model)
    result = extract(solver, y, B, pairs, students)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
