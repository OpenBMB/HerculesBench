from evaluators.evaluator import Evaluator

import torch
from llava_yivl.conversation import conv_templates
from llava_yivl.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava_yivl.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
import os
import json
import re

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def process_image_url(img_path):
    try:
        image = Image.open(img_path)
        return image
    except:
        return None


class YI_VL_Evaluator(Evaluator):
    def __init__(self, model_name, cuda_device_id="0", k=-1):
        super(YI_VL_Evaluator, self).__init__(model_name, k)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id

        disable_torch_init()
        key_info["model_path"] = model_name
        get_model_name_from_path(model_name)
        self.tokenizer, self.vl_model, self.image_processor, self.context_len = load_pretrained_model(model_name)
        self.vl_model = self.vl_model.to(dtype=torch.bfloat16)
    
    def split_markdown(self, md):
        # use regex to extract image property
        items = re.split('(<img_\d+>)', md) # 对输入的数据按照<img_数字>拆分为列表

        # 从分割后的字符串列表中去除空的元素（完全由空白符、换行符或制表符构成的字符串）
        items = [item for item in items if item and not item.isspace()]
        message_items = [] # 对md解析后的，输入给模型的message列表
        images = []
        for item in items:
            if item[:5] == '<img_': # 图片
                image_path = os.path.join(self.image_parent_dir, item[1:-1]+'.jpg') # 图片路径
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpg", ".jpeg")
                    if not os.path.exists(image_path):
                        print('Image file not found!')
                image_pil = process_image_url(image_path)
                if image_pil is not None:
                    images.append(image_pil)
            else:
                message_items.append(item.strip()) # 模型的文本接口格式
        return message_items, images
        
    def make_input(self, prompt, question_content):
        question_message, images = self.split_markdown(prompt + '\n' + question_content)
        if self.is_chinese:
            messages = '请根据要求，完成下面的竞赛题目。' + '\n'.join(question_message)
        else:
            subject = 'Math' if self.is_math else 'Physics'
            messages = 'Please answer the following competition questions as required.' + '\n'.join(question_message)
        messages = DEFAULT_IMAGE_TOKEN + "\n" + messages
        conv = conv_templates["mm_default"].copy()
        conv.append_message(conv.roles[0], messages)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return [prompt, images]


    def get_answer(self, input):
        prompt, images = input
        if len(images) > 1 or len(images) == 0:
            return ""
        else:
            image = images[0]
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        if getattr(self.vl_model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        conv = conv_templates["mm_default"].copy()
        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.vl_model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs