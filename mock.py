import json
import random
from faker import Faker

def generate_mock_data(extra_teacher_prob=0.1):
    fake = Faker()
    
    data = {
        "limits": {
            "MAX_CLASSES_PER_BLOCK": 10,
            "MAX_STUDENTS_PER_CLASS": 15,
            "MIN_STUDENTS_PER_CLASS": 1,
            "MAX_BLOCKS": 40
        },
        "teachers": {},
        "subjects": {},
        "students": {}
    }
    
    # Create teachers for upper school (10 total)
    senior_teachers = [f"teacher_s{i}" for i in range(1, 11)]
    for teacher_id in senior_teachers:
        data["teachers"][teacher_id] = {"name": fake.name()}
    
    # Create subjects for DP1 and DP2
    years = ["DP1", "DP2"]
    subject_types = {
        "simple": {"hours": 3, "count": 6},
        "advanced": {"hours": 5, "count": 6},
        "extra": {"hours": 3, "count": 1}
    }
    
    # Generate subjects
    for year in years:
        for subj_type, config in subject_types.items():
            for i in range(1, config["count"] + 1):
                subject_id = f"{subj_type}_{year}_{i}"
                
                # Pick the main teacher
                main_teacher = random.choice(senior_teachers)
                teachers_list = [main_teacher]
                
                # With 10% probability add a second teacher
                if random.random() < extra_teacher_prob and subj_type != "extra":
                    other_teachers = [t for t in senior_teachers if t != main_teacher]
                    if other_teachers:
                        teachers_list.append(random.choice(other_teachers))
                
                data["subjects"][subject_id] = {
                    "name": f"{subj_type.capitalize()} {year} Subject {i}",
                    "hours": config["hours"],
                    "teachers": teachers_list
                }
    
    # Create students
    student_count = 0
    for year in years:
        for i in range(1, 16):  # 15 students per year level
            student_id = f"student_{year}_{i}"
            student_count += 1
            
            # Choose subjects
            simple_subjects = [f"simple_{year}_{i}" for i in range(1, 7)]
            advanced_subjects = [f"advanced_{year}_{i}" for i in range(1, 7)]
            
            # Randomly select 3 simple and 3 advanced subjects
            selected_simple = random.sample(simple_subjects, 3)
            selected_advanced = random.sample(advanced_subjects, 3)
            extra_subject = [f"extra_{year}_1"]
            
            data["students"][student_id] = {
                "name": fake.name(),
                "subjects": selected_simple + selected_advanced + extra_subject
            }
    
    print(f"Generated mock data with {len(data['teachers'])} teachers, "
          f"{len(data['subjects'])} subjects and {student_count} students")
    return data

if __name__ == "__main__":
    # Generate data with 10% chance of an extra teacher
    mock_data = generate_mock_data(extra_teacher_prob=0.1)
    
    with open("mock_data.json", "w") as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)
    
    print("Mock data written to mock_data.json")
