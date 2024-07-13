# import json
from evaluators.evaluator import Evaluator
from time import sleep
import re, os
# import torch
from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams


class LLaMA3_Evaluator(Evaluator):
	def __init__(self, model_name, cuda_device_id="0", k=-1):
		super(LLaMA3_Evaluator, self).__init__(model_name, k)

		# 指定要使用的 GPU
		os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
		self.model = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=len(cuda_device_id.split(",")))
		self.tokenizer = self.model.get_tokenizer()
		self.sampling_params = SamplingParams(
			temperature=0.1, 
			top_p=0.95, 
			max_tokens=2048,
			stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
			)


	def make_input(self, prompt, question_content):
		if self.is_chinese:
			content = prompt + '\n' + question_content + '\n'
			content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
			messages = [
				{
					'role': 'system',
					'content': f'你是一个中文人工智能助手。请根据要求，完成下面的竞赛题目。'
				},
				{
					'role': 'user',
					'content': content
				}
			]
		else:
			content = prompt + '\n' + question_content + '\n'
			content += 'Please reason step by step, and put your final answer within \\boxed{}.'
			messages = [
				{
					'role': 'system',
					'content': f'You are an AI assistant. Please answer the following competition questions as required.'
				},
				{
					'role': 'user',
					'content': content
				}
			]
		return messages


	def get_answer(self, input):
		input = self.tokenizer.apply_chat_template(input, tokenize=False)
		outputs = self.model.generate(input, self.sampling_params)
		result = outputs[0].outputs[0].text
		return result
