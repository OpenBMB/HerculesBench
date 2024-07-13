import os
import json
from tqdm import tqdm


class Evaluator:
    def __init__(self, model_name, k=-1):
        self.model_name = model_name
        self.k = k
        self.json_dataset_path = ''
        self.image_parent_dir = ''

    def make_prompt(self, question):
        grade = self.grade
        subject = self.subject
        # print(f"Mother fucking json_dataset_path: {self.json_dataset_path}")
        if 'zh_k12' in self.json_dataset_path: # 中文53数据
            box = '\\boxed{选项符号}'
            prompt = f"以下是中国{grade}{subject}考试的选择题。请根据题目的要求，从所提供的选项中选出唯一符合题目要求的正确选项。解答过程中使用的变量和公式请使用LaTex格式表示。请在最后以“所以最终答案是{box}。”的形式，显式给出正确选项的符号。"

        elif 'Linguistics' in self.json_dataset_path:
            box = '\\boxed{YOUR_TRANSLATION_SENTENCE}'
            prompt = f"The following is a translation question from the Internationl Linguistics Olympiad test. Analyze the expression patterns of different languages and find out the right translation. Finally, present the correct translation in the form 'So the final answer is {box}.'"

        elif 'Olympiad' in self.json_dataset_path: # Olympiad数据
            box = '\\boxed{option_symbol}'
            prompt = f"The following is a multiple-choice question from the International {subject} Olympiad competition. Based on the requirements of each question, please select the one correct option from the provided choices that meets the question. Use LaTeX format to represent variables and formulas during the solution process. Finally, explicitly present the correct option symbol in the form 'So the final answer is {box}.'"

        elif 'scienceqa_data' in self.json_dataset_path: # Olympiad数据
            box = '\\boxed{option_symbol}'
            prompt = f"The following is a multiple-choice question from the US {grade} {subject} test. Based on the requirements of each question, please select the one correct option from the provided choices that meets the question. Use LaTeX format to represent variables and formulas during the solution process. Finally, explicitly present the correct option symbol in the form 'So the final answer is {box}.'"
        
        return prompt
    

    def make_input(self, prompt, question_content):
        input = prompt + '\n' + question_content
        return input
    
    def get_answer(self, input):
        pass

    def get_image_mapping_dict(self):
        print(self.json_dataset_path)
        # self.image_parent_dir = os.path.join(os.path.dirname(self.json_dataset_path), 'images')
        self.image_parent_dir = os.path.join(os.path.dirname(os.path.dirname(self.json_dataset_path)), 'images')
        if not os.path.exists(self.image_parent_dir):
            print('Cannot find image directory!')
            exit()
    
    def get_mapping(self, data_path):
        mapping_path = os.path.join(os.path.dirname(os.path.dirname(data_path)),'images/mapping.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)

    def eval_dataset(self, json_dataset_path, json_dataset, save_result_dir):
        self.json_dataset_path = json_dataset_path
        self.is_chinese = True if "zh_data" in json_dataset_path else False
        self.get_mapping(self.json_dataset_path)
        self.get_image_mapping_dict() # Confirm if there is an image folder
        self.grade = json_dataset[0]['step']
        self.subject = json_dataset[0]['subject']


        model_name = self.model_name.split("/")[-1].strip() # For paths like deepseek, extract the model name.

        if not os.path.exists(save_result_dir):
            os.mkdir(save_result_dir)


        if os.path.exists(os.path.join(save_result_dir, f'{model_name}.jsonl')):
            already_done_ids = []
            with open(os.path.join(save_result_dir, f'{model_name}.jsonl'), 'r', encoding='utf-8') as f:
                for line in f:
                    question = json.loads(line)
                    model_output = question["model_output"][list(question["model_output"].keys())[0]]["raw_output"]
                    if model_output == "" or "Requests rate limit exceeded" in model_output:
                        continue
                    already_done_ids.append(question['id'])

            new_dataset = []
            for data in json_dataset:
                if data['id'] not in already_done_ids:
                    new_dataset.append(data)
            
            json_dataset = new_dataset
        


        for id in tqdm(range(len(json_dataset))):
            if id >= len(json_dataset): # 防止超出，然后报错，无法保存最后的结果
                break
            question = json_dataset[id] # 这里直接按列表顺序依次读取。是一个字典
            prompt = self.make_prompt(question)
            content = question['question'] + '\n' + '\n'.join([opt['number'] + opt['symbol'] + opt['content'] for opt in question['options']])
            input = self.make_input(prompt, content)

            answer = self.get_answer(input)
            if 'model_output' not in question.keys():
                question['model_output'] = {model_name:{'raw_output':answer}}
            else:
                question['model_output'][model_name] = {'raw_output':answer}


            with open(os.path.join(save_result_dir, f'{model_name}.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(question, ensure_ascii=False) + '\n')

        print(f'Evaluation finished for {json_dataset_path}.')