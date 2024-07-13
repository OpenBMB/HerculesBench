from evaluators.evaluator import Evaluator
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



class YI_Chat_Evaluator(Evaluator):
    def __init__(self, model_name, cuda_device_id = "0", k=-1):
        super(YI_Chat_Evaluator, self).__init__(model_name, k)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


    def make_input(self, prompt, question_content):
        question_message = prompt + '\n' + question_content
        if self.is_chinese:
            subject = '数学' if self.is_math else '物理'
            messages = '请根据要求，完成下面的{}竞赛题目。'.format(subject) + '\n'.join(question_message)
        else:
            subject = 'Math' if self.is_math else 'Physics'
            messages = 'Please answer the following {} competition questions as required.'.format(subject)  + '\n'.join(question_message)
        messages = [
            {"role": "user", "content": messages}
        ]
        return messages

    def get_answer(self, input):
        input_tensor  = self.tokenizer.apply_chat_template(conversation=input, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_tensor.to('cuda'), max_new_tokens=2048,eos_token_id=7)
        response = self.tokenizer.decode(output_ids[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response