# School Schedule Optimizer

This utility creates a timetable for a small (or maybe even big, if you have computing resources to find the solution) school using [Google OR-Tools](https://developers.google.com/optimization/). It reads a JSON configuration that lists subjects, teachers, students and available rooms, then searches for an arrangement of lessons with the smallest penalty score.

**Positives**
- Flexible data format that covers many real-life constraints
- Generates both machine readable JSON and an easy HTML timetable
- Includes analysis tables for teachers and students

**IMPORTANT**
- Preparing the configuration file takes time
- Tested only on relatively small data sets (up to 50 students); large schools may need fine-tuning
- Searching for the best schedule will take a long time. Pay attention to script output, when it stops changing - probably you found solution which is good enough.

## Quick start

The following steps assume absolutely no prior experience with the command line. For configuration format look below.

### 1. Download the program

1. Open the project page on GitHub and click **Code → Download ZIP**.
2. Unpack the ZIP to a folder such as **Downloads/schedule** (macOS) or **C:\Users\you\Downloads\schedule** (Windows).

### 2. Install Python

Download and install **Python 3** from [python.org](https://www.python.org/downloads/). On Windows check **Add Python to PATH** during setup.

### 3. Install OR-Tools

Open a terminal window:
- **macOS:** open **Finder → Applications → Utilities → Terminal**.
- **Windows:** press **Win+R**, type `cmd` and press **Enter**.

Run the following command:

```bash
pip install ortools
```

### 4. Prepare a configuration file

Inside the project folder make a copy of `config-example.json` and name it `schedule-config.json`.

**Editing on macOS**
1. Open Finder and locate `schedule-config.json`.
2. Right click the file and choose **Duplicate** if you want an extra backup.
3. Edit the file with **TextEdit** or the free [Sublime Text](https://www.sublimetext.com/).
4. Save the file with **⌘S**.

**Editing on Windows**
1. Use File Explorer to open the project folder.
2. Right click `schedule-config.json` and choose **Copy**, then **Paste** to keep a spare copy.
3. Edit the file with **Notepad**, [Notepad++](https://notepad-plus-plus.org/) or [Sublime Text](https://www.sublimetext.com/).
4. Use **File → Save** when done.

### 5. Run the scheduler

1. Open Terminal or Command Prompt as described above.
2. Move to the project directory:
   - macOS example: `cd ~/Downloads/schedule`
   - Windows example: `cd %USERPROFILE%\Downloads\schedule`
3. Check you are in the right place by typing `pwd` on macOS or `cd` on Windows. You should see the path to the folder with `newSchedule.py`.
4. Start the program with the default file names:

```bash
python newSchedule.py
```

Advanced users can provide three optional parameters: a config file, the JSON output file and the HTML timetable.

```bash
python newSchedule.py schedule-config.json schedule.json schedule.html
```

While the solver runs you will see lines similar to

```
#163   1280.39s best:487960 next:[208710,487950] graph_cst_lns (d=9.26e-01 s=10029 t=0.11 p=0.51 stall=181 h=stalling)
```

Press **Ctrl+C** to stop early; the best solution so far will be saved. The search is CPU intensive and may run for hours.

## Configuration overview

The configuration file is divided into several sections. Values are stored as `[value, "description"]` pairs. The **settings**, **defaults** and **penalties** blocks are required. The **model** block is optional and should only be changed if you understand the effects.

### settings (required)
- **objective** – `"total"` minimises the sum of all penalties; `"fair"` tries to distribute penalties evenly. *(default `"total"`)*
- **teacherAsStudents** – how many student opinions one teacher counts as when calculating penalties. *(default 15)*

### defaults (required)
- **teacherImportance** – base weight for teachers. Higher values make teacher gaps more costly. *(default 20)*
- **studentImportance** – base weight for students. Increase to prioritise student convenience. *(default 10)*
- **optimalSlot** – preferred starting slot when a subject lacks its own. `0` is the first lesson. *(default 0)*
- **teacherArriveEarly** – if `true`, teachers are present from the first slot even without a lesson. *(default `false`)*
- **studentArriveEarly** – if `true`, students arrive for slot `0` even when their first lesson is later. *(default `true`)*
- **permutations** – allow subject classes to be arranged in any order. *(default `true`)*
- **avoidConsecutive** – discourage placing the same subject on neighbouring days. *(default `true`)*

### penalties (required)
- **gapTeacher** – penalty for idle teacher slots. High values reduce teacher free time, low values allow it. *(default 30)*
- **gapStudent** – penalty for gaps in a student day. Raise to keep days compact. *(default 100)*
- **unoptimalSlot** – penalty for each slot away from a subject's preferred time. Bigger values keep lessons closer to their optimum. *(default 1)*
- **consecutiveClass** – penalty when the same subject appears on consecutive days. High numbers spread classes apart. *(default 10)*
- **teacherLessonStreak** – array of penalties based on consecutive lessons for teachers. The last value applies to any longer streak.
- **studentLessonStreak** – similar penalties for students based on consecutive lessons.

### days
List the days of teaching and available lesson numbers:
```json
{"name": "Monday", "slots": [0,1,2,3,4]}
```

### cabinets
Rooms where classes may take place. Each entry defines a capacity and optional
list of subjects allowed in that room:
```json
"Room 101": {"capacity": 20, "allowedSubjects": ["Chemistry"]}
```

### subjects
Describe each subject with its teachers and lesson structure.
- `teachers` – teachers who can teach the subject
- `primaryTeachers` – teachers that must be present
- `requiredTeachers` – number of teachers needed for each lesson
- `requiredCabinets` – number of rooms needed for each lesson. When more than one
  cabinet is selected the combined capacity of all chosen rooms must fit the
  class size
 - `optimalSlot` – preferred starting slot
- `classes` – list of lesson lengths that must occur on separate days
- `allowPermutations` – allow the lesson order to change
- `avoidConsecutive` – try not to schedule on back-to-back days
- `cabinets` – rooms where the lesson may take place

### teachers
Mapping of teacher names to optional settings.
- `importance` – overrides the default teacher weight
- `arriveEarly` – whether the teacher is present from slot `0`
- `allowedSlots` – specific slots when the teacher can work
- `forbiddenSlots` – slots the teacher cannot work

### students
List of students with their subjects and optional settings.
- `group` – number of students if this is a group entry
- `importance` – overrides the default student weight
- `arriveEarly` – whether the student arrives from the first slot
- `allowedSlots` – specific slots when the student can attend
- `forbiddenSlots` – slots the student cannot attend

### lessons
Optional fixed lessons specified as `[day, slot, subject, room, [teachers]]` to lock classes to certain times.

### model (optional)
- `maxTime` – solving time limit in seconds (default three hours). A larger value can improve results but takes longer
- `workers` – CPU cores to use. By default it uses available cores minus two, but not less than four
- `showProgress` – `true` to print progress lines during the search (default)

## Using the results

- **schedule.json** – machine readable output for further processing
- **schedule.html** – open in a browser to inspect the timetable with colour coded penalties

## Changelog

### 1.1.0
- Added penalties for consecutive classes and lesson streaks
- Automatic worker detection and improved defaults
- Expanded documentation with step-by-step instructions

### 1.0.0
- Initial schedule generator with JSON configuration and HTML report
- Basic analysis tables for teachers and students
