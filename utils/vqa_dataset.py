import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN, DEFAULT_PURE_CONV

DEFAULT_IMAGE_TOKEN_NEW = "<|vision_start|><|image_pad|><|vision_end>"

def preprocess_multimodal(source):
    """
    Modify conversation entries that contain the default image token.
    For the first sentence that contains the token, replace it with the
    Qwen-specific token.
    """
    for i, sentence in enumerate(source):
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if i == 0:
                sentence["value"] = DEFAULT_IMAGE_TOKEN_NEW + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source

class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
        model_name="qwen_vl", 
    ):
        """
        Initialize the VQADataset with Qwen-specific modifications.
        
        Args:
            base_image_dir (str): Root directory for dataset files.
            tokenizer: Tokenizer used for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): (Not used here, kept for signature compatibility.)
            exclude_val (bool): Whether to exclude validation data.
            vqa_data (str): Name of the VQA JSON file (without extension).
            model_name (str): Identifier for the model ("qwen_vl" or "llava").
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.model_name = model_name.lower()

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        if vqa_data == 'llava_v1_5_mix665k':
            self.vqa_image_root = base_image_dir
        else:
            self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, f"{vqa_data}.json")) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values and pad the tensor image to square dimensions.
        """
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        if 'image' not in item:
            return self.__getitem__(0)
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  

        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        conv = get_default_conv_template(self.model_name).copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Message {j} role mismatch"
            if role == "USER":
                conv.append_message(role, sentence["value"] + DEFAULT_PURE_CONV)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations 

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        return (
            image_path,
            image,
            image_vlm,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
