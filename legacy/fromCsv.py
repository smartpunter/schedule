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
    
    # Track unique identifiers
    teacher_ids = set()
    student_ids = set()
    
    required = ['Year group', 'Subject', 'Weekly hours', 'Teacher', 'Students']
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line_num, row in enumerate(reader, start=2):
            if not row or all((v is None or v == '') for v in row.values()):
                print(f"Skipping empty line {line_num}", file=sys.stderr)
                continue

            if any(field not in row or row[field] in (None, '') for field in required):
                print(f"Skipping invalid line {line_num}: missing data", file=sys.stderr)
                continue

            year_group = row['Year group']
            subject_name = f"{year_group}_{row['Subject']}"

            try:
                hours = int(row['Weekly hours'])
            except (TypeError, ValueError):
                print(f"Skipping invalid line {line_num}: bad hours '{row['Weekly hours']}'", file=sys.stderr)
                continue

            teacher_name = row['Teacher']
            student_names = [name.strip() for name in row['Students'].split(',') if name.strip()]
            
            # Handle teacher
            teacher_id = normalize_id(teacher_name)
            if teacher_id not in teacher_ids:
                config["teachers"][teacher_id] = {"name": teacher_name}
                teacher_ids.add(teacher_id)
            
            # Create a unique subject ID
            subject_id = normalize_id(subject_name)
            if subject_id in config["subjects"]:
                # If the subject already exists, append a suffix
                suffix = 2
                while f"{subject_id}_{suffix}" in config["subjects"]:
                    suffix += 1
                subject_id = f"{subject_id}_{suffix}"
            
            # Add the subject
            config["subjects"][subject_id] = {
                "hours": hours,
                "teachers": [teacher_id]
            }
            
            # Handle students
            for student_name in student_names:
                student_id = normalize_id(student_name)
                if student_id not in student_ids:
                    config["students"][student_id] = {"subjects": []}
                    student_ids.add(student_id)
                
                # Add the subject to the student's list
                if subject_id not in config["students"][student_id]["subjects"]:
                    config["students"][student_id]["subjects"].append(subject_id)
    
    return config

def main():
    input_file = 'data.csv'
    output_file = 'config.json'
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        config = csv_to_config(input_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted: {input_file} -> {output_file}")
        print(f"Teachers: {len(config['teachers'])}")
        print(f"Subjects: {len(config['subjects'])}")
        print(f"Students: {len(config['students'])}")
        print(f"Max blocks: {config['limits']['MAX_BLOCKS']}")
    
    except FileNotFoundError:
        print(f"Error: File not found - {input_file}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required column - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

