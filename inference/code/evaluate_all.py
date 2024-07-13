import os
import json
import argparse
from evaluators.gpt_4v import GPT_4V_Evaluator
from evaluators.gemini_pro_vision import Gemini_Pro_Vision_Evaluator
from evaluators.qwen_vl import Qwen_VL_Evaluator
# from evaluators.deepseek_evaluator import Deepseek_Evaluator
from evaluators.check_prompt import Check_Prompt_Evaluator
from evaluators.gpt_4o import GPT_4o_Evaluator
from evaluators.claude3v_opus import Claude3v_Opus_Evaluator
# from evaluators.llama3_evaluator import LLaMA3_Evaluator
# from evaluators.yi_vl import YI_VL_Evaluator
# from evaluators.yi_chat import YI_Chat_Evaluator
# from evaluators.llava_evaluator import LLaVa_Evaluator


def main(args):
	if 'gpt-4o' in args.model_name: # GPT-4o
		evaluator = GPT_4o_Evaluator(
			model_name=args.model_name
		)
	elif 'gpt-4' in args.model_name: # GPT-4V
		evaluator = GPT_4V_Evaluator(
			model_name=args.model_name
		)
	elif 'claude3v-opus' in args.model_name: # claude3v-opus
		evaluator = Claude3v_Opus_Evaluator(
			model_name = args.model_name
		)
	elif 'qwen-vl' in args.model_name:
		evaluator = Qwen_VL_Evaluator(
			model_name=args.model_name
		)
	elif 'gemini-pro' in args.model_name:
		evaluator = Gemini_Pro_Vision_Evaluator(
			model_name=args.model_name,
			api_key=args.api_key
		)
	elif 'Llama-3' in args.model_name:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = LLaMA3_Evaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
	elif 'Yi-VL' in args.model_name:
		evaluator = YI_VL_Evaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
		args.model_name = 'Yi-VL'
	elif 'Yi-34B-Chat' in args.model_name:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = YI_Chat_Evaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
		args.model_name = 'Yi-34B-Chat'
	elif 'Nous-Hermes-2-Yi-34B' in args.model_name:
		assert 'TO' in args.dataset_name	# text-only model
		evaluator = YI_Chat_Evaluator(
			model_name=args.model_name,
			cuda_device_id=args.cuda_device
		)
		args.model_name = 'Nous-Hermes-2-Yi-34B'
	
	elif 'llava-v1.6-34b' in args.model_name:
		evaluator = LLaVa_Evaluator(
			model_name = args.model_name,
			cuda_device_id=args.cuda_device
		)
		args.model_name = 'llava-v1.6-34b'

	elif 'check-prompt' in args.model_name:
		evaluator = Check_Prompt_Evaluator(
			model_name=args.model_name
		)
	else:
		print("Unknown model name")
		exit()
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)
	if not os.path.exists(dataset_save_dir):
		os.mkdir(dataset_save_dir)
	# dataset_path = os.path.join('data_0301', args.dataset_name)
	if not args.saving_name:
		save_result_dir = os.path.join(dataset_save_dir, args.model_name)
	else:
		save_result_dir=os.path.join(dataset_save_dir, args.saving_name)

	print("dataset_path: ", args.dataset_path)
	print("save_result_dir: ", save_result_dir)
	with open(args.dataset_path, 'r', encoding='utf-8') as f:
		json_dataset = json.load(f)
		evaluator.eval_dataset(
			json_dataset_path=args.dataset_path,
			json_dataset=json_dataset,
			save_result_dir=save_result_dir
		)
			

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, required=True)
	parser.add_argument("--dataset_path", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default='../generated')
	parser.add_argument("--saving_name", type=str)
	parser.add_argument("--cuda_device", type=str)
	parser.add_argument("--api_key", type=str)
	args = parser.parse_args()
	args.dataset_name = args.dataset_path.split("/")[-1].strip()[:-5]
	main(args)