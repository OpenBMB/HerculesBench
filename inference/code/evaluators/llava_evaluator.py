import torch
from evaluators.evaluator import Evaluator
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


class LLaVa_Evaluator(Evaluator):
    def __init__(self, model_name, cuda_device_id="0", k=-1):
        super(LLaVa_Evaluator, self).__init__(model_name, k)

        disable_torch_init()
        self.model_path = model_name
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, None, self.model_name
        ) # 记载模型，可以放在init里面
    
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
                images.append(image_path)
            else:
                message_items.append(item.strip()) # 模型的文本接口格式
        return message_items, images
    
    def make_input(self, prompt, question_content):
        question_message, images = self.split_markdown(prompt + '\n' + question_content)
        if self.is_chinese:
            messages = '请根据要求，完成下面的竞赛题目。' + '\n'.join(question_message)
        else:
            messages = 'Please answer the following competition questions as required.' + '\n'.join(question_message)

        return [messages, images]
    
    def get_answer(self, input):
        query, images = input
        if len(images) > 1 or len(images) == 0:
            return ""
        else:
            image = images[0]
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 以上是文本
        # 下面是图片
        image_files = images
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs