# Schedule Generator

This repository contains two scripts for experimenting with generating class schedules using [Google OR‑Tools](https://developers.google.com/optimization/).

* **`mock.py`** – builds a sample configuration JSON with fake teachers, subjects and students using the `Faker` library.
* **`solver.py`** – reads a configuration file and computes an optimal allocation of classes to blocks.

## Requirements

Install the minimal dependencies with:

```bash
pip install ortools Faker
```

## Generating Example Data

Run the mock data generator to produce a configuration file:

```bash
python mock.py
```

A file called `mock_data.json` will be created in the repository directory.

## Running the Solver

Run the solver with the configuration JSON:

```bash
python solver.py mock_data.json
```

The schedule will be written to `schedule.json`. It contains an array of blocks. Each block is an array of objects describing a class with the following fields:

- `subject` – subject identifier
- `teacher` – teacher identifier
- `students` – list of student IDs attending that class

## Configuration Format

A configuration JSON consists of four top‑level keys: `limits`, `teachers`, `subjects` and `students`. Below is a small example illustrating the structure:

```json
{
  "limits": {
    "MAX_CLASSES_PER_BLOCK": 10,
    "MAX_STUDENTS_PER_CLASS": 15,
    "MIN_STUDENTS_PER_CLASS": 1,
    "MAX_BLOCKS": 40
  },
  "teachers": {
    "teacher_a": {"name": "Alice"}
  },
  "subjects": {
    "math_DP1": {
      "name": "Math DP1",
      "hours": 5,
      "teachers": ["teacher_a"]
    }
  },
  "students": {
    "student_1": {
      "name": "Bob",
      "subjects": ["math_DP1"]
    }
  }
}
```

`mock.py` generates a much larger file with the same shape. You can edit any field or create your own configuration following this format.
