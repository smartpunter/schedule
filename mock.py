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
    
    # Создаем учителей (10 для старших классов)
    senior_teachers = [f"teacher_s{i}" for i in range(1, 11)]
    for teacher_id in senior_teachers:
        data["teachers"][teacher_id] = {"name": fake.name()}
    
    # Создаем предметы для DP1 и DP2
    years = ["DP1", "DP2"]
    subject_types = {
        "simple": {"hours": 3, "count": 6},
        "advanced": {"hours": 5, "count": 6},
        "extra": {"hours": 3, "count": 1}
    }
    
    # Генерируем предметы
    for year in years:
        for subj_type, config in subject_types.items():
            for i in range(1, config["count"] + 1):
                subject_id = f"{subj_type}_{year}_{i}"
                
                # Выбираем основного учителя
                main_teacher = random.choice(senior_teachers)
                teachers_list = [main_teacher]
                
                # С вероятностью 10% добавляем второго учителя
                if random.random() < extra_teacher_prob and subj_type != "extra":
                    other_teachers = [t for t in senior_teachers if t != main_teacher]
                    if other_teachers:
                        teachers_list.append(random.choice(other_teachers))
                
                data["subjects"][subject_id] = {
                    "name": f"{subj_type.capitalize()} {year} Subject {i}",
                    "hours": config["hours"],
                    "teachers": teachers_list
                }
    
    # Создаем учеников
    student_count = 0
    for year in years:
        for i in range(1, 16):  # 15 учеников в каждом классе
            student_id = f"student_{year}_{i}"
            student_count += 1
            
            # Выбираем предметы
            simple_subjects = [f"simple_{year}_{i}" for i in range(1, 7)]
            advanced_subjects = [f"advanced_{year}_{i}" for i in range(1, 7)]
            
            # Случайный выбор 3 простых и 3 сложных предметов
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
    # Генерируем данные с вероятностью дополнительного учителя 10%
    mock_data = generate_mock_data(extra_teacher_prob=0.1)
    
    with open("mock_data.json", "w") as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)
    
    print("Mock данные успешно сохранены в mock_data.json")
