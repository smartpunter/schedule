import json
from collections import defaultdict
from statistics import mean


def load_schedule(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyse_teachers(schedule):
    teachers = defaultdict(lambda: {
        'blocks': 0,
        'students': [],
        'subjects': defaultdict(list)
    })
    for block in schedule:
        for cls in block:
            t = cls['teacher']
            s = cls['subject']
            n = len(cls['students'])
            teachers[t]['blocks'] += 1
            teachers[t]['students'].append(n)
            teachers[t]['subjects'][s].append(n)
    return teachers


def analyse_students(schedule):
    student_hours = defaultdict(lambda: defaultdict(int))
    for block in schedule:
        for cls in block:
            subj = cls['subject']
            for stu in cls['students']:
                student_hours[stu][subj] += 1
    return student_hours


def analyse_subjects(schedule):
    subj_students = defaultdict(set)
    for block in schedule:
        for cls in block:
            subj = cls['subject']
            subj_students[subj].update(cls['students'])
    return subj_students


def report_teachers(teachers):
    print('=== Teachers ===')
    sorted_teachers = sorted(teachers.items(), key=lambda x: x[1]['blocks'], reverse=True)
    for tid, info in sorted_teachers:
        avg = mean(info['students']) if info['students'] else 0
        print(f"{tid}: {info['blocks']} блоков, {avg:.1f} ученик на класс")
        subj_stats = sorted(info['subjects'].items(), key=lambda x: len(x[1]), reverse=True)
        for subj, counts in subj_stats:
            avg_s = mean(counts) if counts else 0
            print(f"{subj}: {len(counts)} блоков, {avg_s:.1f} ученика на класс")
        print()


def report_students(student_hours):
    print('=== Students ===')
    for sid, subj_map in sorted(student_hours.items()):
        print(sid)
        parts = [f"{subj}: {hours}" for subj, hours in sorted(subj_map.items())]
        print('; '.join(parts))
        print()


def report_subjects(subj_students):
    print('=== Subjects ===')
    sorted_subj = sorted(subj_students.items(), key=lambda x: len(x[1]), reverse=True)
    for subj, students in sorted_subj:
        print(f"{subj}: {len(students)} students")
        print()


def main():
    schedule = load_schedule('schedule.json')
    teachers = analyse_teachers(schedule)
    student_hours = analyse_students(schedule)
    subj_students = analyse_subjects(schedule)

    report_teachers(teachers)
    report_students(student_hours)
    report_subjects(subj_students)


if __name__ == '__main__':
    main()
