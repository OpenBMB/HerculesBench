import json, os, re
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
from sympy import rotations
import regex

def extract_boxed_content(text):
    # 使用递归正则表达式匹配 \boxed{...}，其中 ... 可能包含任意多层嵌套的 {}
    pattern = r'\\boxed{((?:[^{}]|(?R))*)}'
    results = regex.findall(pattern, text)

    if results:
        return results[0]
    
    return None


def extract_first_letter(s):
    # 使用正则表达式查找符合 \\boxed{SUBSTRING} 格式的子字符串
    # substring = extract_boxed_content(s)
    # print(substring)
    substring = s.split("\\boxed{")[1] if "\\boxed{" in s else s
    

    if substring:
        substring = re.sub(r'\\text{(.*?)}', r'\1', substring)

        # 提取子字符串中的第一个字母
        for char in substring:
            if char.isupper():
                return char
        
    return None


root_path = "OlympiadBench-main/inference/generated"

results = {}

subjects = os.listdir(root_path)
subjects.sort()

for subject in subjects:
    subject_path = os.path.join(root_path, subject)
    for model in os.listdir(subject_path):
        model_name = model if model != "claude-opus" else "claude3v-opus"
        model_name = "gemini-pro" if model_name == "gemini-1.5-pro" else model_name
        model_output_path = os.path.join(subject_path, model, f"{model_name}.jsonl")

        if not os.path.exists(model_output_path):
            continue

        dataset = []
        with open(model_output_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        
        # Compute the score
        right_num = 0
        total_num = len(dataset)
        nan_num = 0
        for data in dataset:
            if data["model_output"][model_name]["raw_output"] == "":
                nan_num += 1
                continue
            model_output = extract_first_letter(data["model_output"][model_name]["raw_output"])
            # if model == "gemini-1.5-pro":
            #     print(model_output, data["model_output"][model_name]["raw_output"][-20:])
            #     a = input()
            if model_output == data["answer"]["answer"]:
                right_num += 1
        
        total_num -= nan_num
        
        score = (float(right_num) / total_num) * 100
        score = round(score, 1)

        print(f"{model_name:<20}________{subject:<30}_________score: {score}%")


        model_name = "gemini-1.5-pro" if model == "gemini-1.5-pro" else model_name
        if model_name not in results:
            results[model_name] = {}
        
        if subject.startswith("science_qa_split"):
            results[model_name][subject] = score
            continue
        
        results[model_name][subject.split("_")[0]] = score

print(results)


# 转换数据为 DataFrame
df = pd.DataFrame(results).transpose()
zh_font = fm.FontProperties(fname='OlympiadBench-main/ukai.ttc')

# 创建图形和轴
plt.figure(figsize=(15, 10))

# 使用高对比度的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 绘制每个模型的成绩
for idx, model in enumerate(df.index):
    plt.plot(df.columns, df.loc[model], marker='o', label=model, color=colors[idx])

# 添加标题和标签
plt.title('Performance Comparison of Different Models')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.legend(title='Models')
plt.grid(True)
plt.xticks(rotation=45, fontproperties=zh_font)
plt.tight_layout()

# 显示图形
plt.show()