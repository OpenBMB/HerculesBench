import json, os, random

path = "OlympiadBench-main/dataset/zh_data/data"

grade_dict = []

none_num = 0

step_grade_list = {
    "小学": [
        "一年级",
        "二年级",
        "三年级",
        "四年级",
        "五年级",
        "六年级"
    ],
    "初中": [
        "七年级",
        "八年级",
        "九年级"
    ],
    "高中": [
        "高一",
        "高二",
        "高三"
    ]
}


for file in os.listdir(path):
    file_path = os.path.join(path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    grade_dict = {
        "一年级": [],
        "二年级": [],
        "三年级": [],
        "四年级": [],
        "五年级": [],
        "六年级": [],
        "七年级": [],
        "八年级": [],
        "九年级": [],
        "高一": [],
        "高二": [],
        "高三": []
    }

    for data in dataset:
        grade = data["grade"]
        if data["grade"] == "":
            grade = random.choice(step_grade_list[data["step"]])
        
        grade_dict[grade].append(data)

    
    for grade in grade_dict:
        print(len(grade_dict[grade]))
    
    print("="*70 + f"The above is {file}" + "="*70)


            
# print(grade_dict)
# print(none_num)