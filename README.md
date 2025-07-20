# School Schedule Optimizer

`newSchedule.py` generates an optimal timetable using [Google OR‑Tools](https://developers.google.com/optimization/). The script reads a JSON configuration that describes rooms, teachers, students and subjects, then tries to create the best possible schedule.

## Quick start

### 1. Install Python

Download and install **Python 3** from [python.org](https://www.python.org/downloads/) if it is not already available. During installation on Windows check the option *Add Python to PATH*.

### 2. Install the required package

Open **Terminal** on macOS or **Command Prompt** on Windows and install the solver library:

```bash
pip install ortools
```

### 3. Prepare a configuration file

Copy `config-example.json` to `schedule-config.json` and adjust it to match your school. The example shows the structure but is intentionally incomplete. Important blocks are:

- `settings` – global parameters such as allowed consecutive lessons.
- `penalties` – weights used when evaluating schedule quality.
- `days` – list of days and available lesson slots.
- `cabinets` – available rooms with capacities.
- `subjects`, `teachers`, `students` – information about who studies or teaches what.
- `model` – solver parameters. `objective` controls optimisation strategy:
  - `total` (default) minimises the sum of all penalties.
  - `fair` minimises the largest individual penalty among teachers and students.

### 4. Run the scheduler

Execute the script and pass the configuration path. The second argument is where to store the result:

```bash
python newSchedule.py schedule-config.json schedule.json
```

After solving, `schedule.json` contains the raw schedule. An interactive HTML view `schedule.html` is also generated so you can browse the timetable in a browser.

## Using the results

- **schedule.json** – structured data that can be post‑processed or imported elsewhere.
- **schedule.html** – open this file in your browser to inspect the timetable. It highlights teachers, students and empty slots.
- The console output provides a textual summary and optional analysis of teachers, students and subjects.

## Development notes

The optimizer was created iteratively while exploring OR‑Tools capabilities. The configuration format was designed to be flexible, allowing arbitrary sets of days, rooms and lesson lengths. `newSchedule.py` also generates human‑readable reports to help understand the computed schedule.

**Strengths**

- Flexible JSON configuration with sensible defaults.
- Produces both machine‑readable and visual outputs.
- Includes analysis tables for teachers, students and subjects.

**Weaknesses**

- Configuration file is quite verbose and requires careful preparation.
- Only lightly tested and may perform slowly on very large data sets.
- The example configuration is intentionally incomplete and must be edited before use.

Whole project was coded using openAI Codex tool. I do not know neither python, nor optimization models. Still, this thing works extremely well for small private school and this is miracle.
