Это проект для оптимизации расписания школы.

Все комментарии в коде пиши на английском.

ВНИМАНИЕ! Не нужно читать дальше, всегда работай только с файлом newSchedule.py, улучшения ещё не закончены!


Проект полностью функционален, но требует полного рефакторинга для улучшения скорости для больших школ.

Основная часть находится в функции build_model файла newSchedule.py, остальное это генерация интерфейса.
В файле config-example.json находится вся информация о доступных настройках и ограничениях модели.

При запросах об оптимизации модели проанализируй логику работы функции build_model, трогать остальные части скрипта нет необходимости, там работа с уже сгенерированными данными.



# Guidelines for Rewriting the Timetabling Project


## 0. High‑level Goals

1. **Shrink the decision space** by replacing thousands of BoolVars with compact interval‑based constructs.
2. **Accelerate solve times** through smarter preprocessing and a staged search (feasible → optimal).
3. **Keep the codebase testable & maintainable** by strictly separating data, model, and search layers.

---

## 1. Directory/Layout Skeleton

Оптимизированный код будет находиться в папке "fast", он будет разбит на несколько файлов и один оркестратор doSchedule.py находящийся в корне, который их подключает и проходит весь цикл:
1. Сначала препроцесснгом обрабатывается файл конфигурации, проводятся проверки и оптимизации.
2. Затем запускается простой солвер, без оптимизации, который находит стартовое решение и убеждается, что оно существует.
2. Затем строится и запускается модель с оптимизациями всех требований, генерирующая оптимальное расписание.
3. После на основании этого расписания генерируется html интерфейс и выводятся отчёты в консоль.

---

## 2. Pre‑processing Module (`preprocessing.py`)  (Rec 5)

| Task    | Detail                                                                                                                                                                  |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.1** | Parse raw CSV/JSON of classes, teachers, rooms, availability.                                                                                                           |
| **2.2** | *Feasible domain filtering*: For each lesson compute allowed `(day, slot)` pairs based on teacher/room availability & school calendar. Remove impossible options early. |
| **2.3** | Map teachers, student groups, cabinets to **contiguous integer IDs**. Export to `namedtuple`/`pydantic` objects for type safety.                                        |
| **2.4** | Serialize the filtered domain (`.pkl` or in‑memory) for quick reuse.                                                                                                    |

---

## 3. Model Construction (`model.py`)

### 3.1 Interval Creation (Rec 1)

```python
# pseudocode
for cls in classes:
    start = model.NewIntVarFromDomain(cls.domain, f"start_{cls.id}")
    interval = model.NewOptionalIntervalVar(start, cls.duration, start + cls.duration,
                                            presence_literal=model.NewBoolVar(f"presence_{cls.id}"),
                                            name=f"lesson_{cls.id}")
```

*Store the `interval` and its `start` for later constraints and penalties.*

### 3.2 Resource Constraints (Rec 2 & 7)

| Resource                     | Technique                                            |
| ---------------------------- | ---------------------------------------------------- |
| **Teacher (capacity 1)**     | `model.AddNoOverlap(teacher_intervals[id])`          |
| **Cabinet (capacity ≥1)**    | `model.AddCumulative(room_tasks, demands, capacity)` |
| **Student group (optional)** | `AddNoOverlap` (if exclusivity needed).              |

Make helper builders: `add_teacher_constraints()`, `add_room_constraints()`.

### 3.3 Auxiliary Day IntVars (Rec 3)

```python
slots_per_day = constants.SLOTS_PER_DAY
# day_i = start_i // slots_per_day
model.AddDivisionEquality(day_var, start_var, slots_per_day)
```

Use `day_var` to express *consecutive‑days* rules: `model.Add(day_j - day_i != 1)` etc.

### 3.4 Penalties & Gaps via Arithmetic (Rec 4)

*Example*: Optimal start slot penalty.

```python
diff = model.NewIntVar(-max_slot, max_slot, f"diff_{cls.id}")
model.Add(diff == start - cls.optimal_slot)
abs_diff = model.NewIntVar(0, max_slot, f"abs_diff_{cls.id}")
model.AddAbsEquality(abs_diff, diff)
model.AddWeightedSum([abs_diff], [weight], total_penalty)
```

Apply similar patterns for **gap length**, **streak length**, etc., by sorting a day’s intervals and using `StartExpr()`/`EndExpr()` differences.

---

## 4. Two‑Phase Search (`search.py`) (Rec 6)

| Phase       | Settings                                                                                                                                                       | Purpose                                                   |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Phase 1** | `model.Minimize(0)` (ignore soft) <br> decision builder: `MODEL_FIRST_SOLUTION`                                                                                | Rapidly obtain *any* feasible timetable.                  |
| **Phase 2** | Unfreeze intervals; minimize total penalty expr. <br> Use **Large Neighborhood Search** with fragments: \* all lessons of a teacher, \* a day’s schedule, etc. | Improve objective without restarting search from scratch. |

Implement CLI flags `--phase1` / `--phase2` for experimentation.

---

## 5. Coding Standards & CI

* **English‑only comments & docstrings** (matches repo rules).
* Type‑annotate all public functions; run `mypy` in CI.
* Unit tests for each helper; E2E test that solves a mini‑school instance < 1 s.
* Default black & isort formatting; pre‑commit hooks.

---

## 6. Task Breakdown by Role

| Role                   | Responsibilities                                           |
| ---------------------- | ---------------------------------------------------------- |
| **Data Engineer**      | Section 2 – build preprocessing pipeline & fixtures.       |
| **Constraint Modeler** | Section 3 – interval creation & resource constraints.      |
| **Penalty Designer**   | Section 3.4 – translate all soft rules into arithmetic.    |
| **Search Tuner**       | Section 4 – parameterise LNS, evaluate runtime vs quality. |
| **Dev Ops**            | Section 5 – CI, linting, container images for solver runs. |

---

