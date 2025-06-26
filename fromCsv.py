import csv
import json
import sys
import re
from collections import defaultdict

def normalize_id(name):
    """Convert name to ID format (replace spaces with underscores)"""
    return re.sub(r'\s+', '_', name.strip())

def csv_to_config(input_file):
    config = {
        "limits": {
            "MAX_STUDENTS_PER_CLASS": 15,
            "MIN_STUDENTS_PER_CLASS": 1,
            "MAX_CLASSES_PER_BLOCK": 10,
            "MAX_BLOCKS": 40
        },
        "teachers": {},
        "subjects": {},
        "students": {}
    }
    
    # Для отслеживания уникальных идентификаторов
    teacher_ids = set()
    student_ids = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            year_group = row['Year group']
            subject_name = f"{year_group}_{row['Subject']}"
            hours = int(row['Weekly hours'])
            teacher_name = row['Teacher']
            student_names = [name.strip() for name in row['Students'].split(',')]
            
            # Обработка учителя
            teacher_id = normalize_id(teacher_name)
            if teacher_id not in teacher_ids:
                config["teachers"][teacher_id] = {"name": teacher_name}
                teacher_ids.add(teacher_id)
            
            # Создаем уникальный ID для предмета
            subject_id = normalize_id(subject_name)
            if subject_id in config["subjects"]:
                # Если предмет уже существует, добавляем суффикс
                suffix = 2
                while f"{subject_id}_{suffix}" in config["subjects"]:
                    suffix += 1
                subject_id = f"{subject_id}_{suffix}"
            
            # Добавляем предмет
            config["subjects"][subject_id] = {
                "hours": hours,
                "teachers": [teacher_id]
            }
            
            # Обработка студентов
            for student_name in student_names:
                student_id = normalize_id(student_name)
                if student_id not in student_ids:
                    config["students"][student_id] = {"subjects": []}
                    student_ids.add(student_id)
                
                # Добавляем предмет в список студента
                if subject_id not in config["students"][student_id]["subjects"]:
                    config["students"][student_id]["subjects"].append(subject_id)
    
    return config

def main():
    input_file = 'data.csv'
    output_file = 'config.json'
    
    # Обработка аргументов командной строки
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        config = csv_to_config(input_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Успешно сконвертировано: {input_file} -> {output_file}")
        print(f"Учителей: {len(config['teachers'])}")
        print(f"Предметов: {len(config['subjects'])}")
        print(f"Студентов: {len(config['students'])}")
        print(f"Макс. блоков: {config['limits']['MAX_BLOCKS']}")
    
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден - {input_file}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Ошибка: Отсутствует обязательная колонка - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

