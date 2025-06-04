import os
import random
import cv2
import numpy as np
import torch
import re
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

from datasets import Dataset as HFDataset
from .conversation import get_default_conv_template
from model.segment_anything.utils.transforms import ResizeLongestSide


class COTDataset(torch.utils.data.Dataset):
    """
    COT segmented dataset loader. Loads specified .arrow files from the seg++ directory.
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir: str,
        tokenizer,
        dataset_type: str = "caption||conversation||cot||instance_seg",
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        model_name: str = "qwen_vl",
        image_subdir: str = "train2017",
    ):
        self.base_image_dir = base_image_dir
        self.DATA_DIR = os.path.join(base_image_dir, "seg++")
        self.image_dir = os.path.join(base_image_dir, "coco", image_subdir)
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.model_name = model_name.lower()

        # Parse dataset types
        self.dataset_types = [t.strip().lower() for t in dataset_type.split("||")]
        self.data_by_name = {}
        self.lengths_by_name = {}

        # Load each specified dataset by .arrow file
        for name in self.dataset_types:
            arrow_fp = os.path.join(self.DATA_DIR, f"{name}.arrow")
            if not os.path.exists(arrow_fp):
                continue
            ds = HFDataset.from_file(arrow_fp)
            self.data_by_name[name] = ds
            self.lengths_by_name[name] = len(ds)

        if not self.data_by_name:
            raise RuntimeError(
                f"No valid .arrow datasets found under {self.DATA_DIR} matching types {self.dataset_types}"
            )

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        pad_h = self.img_size - h
        pad_w = self.img_size - w
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _get_image_path(self, item, dataset_name):
        low = dataset_name.lower()
        if any(key in low for key in ("caption", "instance_seg", "cot")):
            return os.path.join(self.image_dir, item.get("image_path", item.get("img_path", "")))
        if "conversation" in low:
            return os.path.join(self.image_dir, item.get("img_pth", ""))
        return None

    def _parse_instances(self, item, dataset_name):
        low = dataset_name.lower()
        caption_cot_pattern = re.compile(r'《\s*(\d+)\s*\|\s*([^》]+?)\s*》')
        if any(key in low for key in ("caption", "cot")):
            answer = item.get("English Answer", "")
            return caption_cot_pattern.findall(answer)
        if "conversation" in low:
            conv_pattern = re.compile(r'<\s*(\d+)\s*\|\s*([^>]+?)\s*>')
            answer = item.get("output", "")
            return conv_pattern.findall(answer)
        if "instance_seg" in low:
            parts = item.get("English Answer", "").split(";")
            insts = []
            for part in parts:
                m = re.match(r'instance id is (\d+), label name is (.+)', part.strip())
                if m:
                    insts.append((m.group(1), m.group(2)))
            return insts
        return []

    def _generate_conversations(self, item, instances, dataset_name, h, w):
        low = dataset_name.lower()

        def _fill_from_polys(polys):
            m = np.zeros((h, w), dtype=np.uint8)
            if not isinstance(polys[0][0], (list, tuple, np.ndarray)):
                polys = [polys]
            for poly in polys:
                if len(poly) < 3:
                    continue
                arr = (np.asarray(poly, dtype=np.float32)
                       .round().astype(np.int32)
                       .reshape(-1, 1, 2))
                cv2.fillPoly(m, [arr], 1)
            return m

        if "instance_seg" in low:
            conversations, masks = [], []
            for inst_id, class_name in instances:
                question = f"What is the {class_name} in this image? Please output segmentation mask."
                answer = "[SEG]"
                conv = get_default_conv_template(self.model_name).copy()
                user_msg = f"<|vision_start|><|image_pad|><|vision_end|>\n{question}"
                conv.append_message(conv.roles[0], user_msg)
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())

                rle_info = item.get("info", {}).get(inst_id)
                if item.get("is_rle") and isinstance(rle_info, dict) and "counts" in rle_info:
                    if not isinstance(rle_info["counts"], bytes):
                        rle_info["counts"] = rle_info["counts"].encode()
                    mask = mask_utils.decode(rle_info)
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask.astype(np.uint8))
                else:
                    polys = item.get("info", {}).get(inst_id)
                    if polys is not None:
                        masks.append(_fill_from_polys(polys))
                # If no valid mask, skip adding to masks list

            return conversations, masks

        if any(key in low for key in ("caption", "cot")):
            question = item.get("English Question", "")
            answer = item.get("English Answer", "")
            inner_pattern = r'《(\d+)\|([^》]+)》'
            outer_pattern = r'\[((?:《\d+\|[^》]+》\s*)+)\]'
            ids = [g[0] for g in re.findall(inner_pattern, answer)]

            def expand_group(match):
                group_text = match.group(1)
                count = len(re.findall(inner_pattern, group_text))
                return " ".join(["[SEG]"] * count)

            tmp = re.sub(outer_pattern, expand_group, answer)
            processed = re.sub(inner_pattern, "[SEG]", tmp)
            conv = get_default_conv_template(self.model_name).copy()
            user_msg = f"<|vision_start|><|image_pad|><|vision_end|>\n{question}"
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], processed)

            masks = []
            for inst_id in ids:
                rle_info = item.get("info", {}).get(inst_id)
                if item.get("is_rle") and isinstance(rle_info, dict) and "counts" in rle_info:
                    if not isinstance(rle_info["counts"], bytes):
                        rle_info["counts"] = rle_info["counts"].encode()
                    mask = mask_utils.decode(rle_info)
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask.astype(np.uint8))
                else:
                    polys = item.get("info", {}).get(inst_id)
                    if polys is not None:
                        masks.append(_fill_from_polys(polys))

            return [conv.get_prompt()], masks

        if "conversation" in low:
            output = item.get("output", "")
            inner_pattern = r'<(\d+)\|([^>]+)>'
            outer_pattern = r'\[((?:<\d+\|[^>]+>\s*)+)\]'
            ids = [g[0] for g in re.findall(inner_pattern, output)]

            def expand_group(match):
                group_text = match.group(1)
                count = len(re.findall(inner_pattern, group_text))
                return " ".join(["[SEG]"] * count)

            tmp = re.sub(outer_pattern, expand_group, output)
            processed = re.sub(inner_pattern, "[SEG]", tmp)
            lines = processed.split("\n")
            conv = get_default_conv_template(self.model_name).copy()
            turn_re = re.compile(r'^(<person>|<robot>):\s*(.*)$')
            seen_first_user = False

            for ln in lines:
                m = turn_re.match(ln)
                if not m:
                    continue
                speaker_tag, utter = m.groups()
                if speaker_tag == "<person>":
                    role = conv.roles[0]
                    if not seen_first_user:
                        utter = f"<|vision_start|><|image_pad|><|vision_end|>\n{utter}"
                        seen_first_user = True
                else:
                    role = conv.roles[1]
                conv.append_message(role, utter)

            masks = []
            for inst_id in ids:
                ann = item.get("info", {}).get(inst_id)
                if ann is None:
                    continue
                polys = ann.get("polygon")
                if polys is None:
                    raw = ann.get("polygon_raw", [])
                    polys = [
                        [[int(round(x)), int(round(y))] for x, y in poly]
                        for poly in raw
                    ]
                masks.append(_fill_from_polys(polys))

            return [conv.get_prompt()], masks

        return [], []

    def __getitem__(self, idx):
        dataset_name = random.choice(list(self.data_by_name.keys()))
        ds = self.data_by_name[dataset_name]
        i = random.randint(0, self.lengths_by_name[dataset_name] - 1)
        item = ds[i]

        img_path = self._get_image_path(item, dataset_name)
        if not img_path or not os.path.exists(img_path):
            return self.__getitem__(idx)

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__(idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        instances = self._parse_instances(item, dataset_name)
        if not instances:
            return self.__getitem__(idx)
        if len(instances) > self.num_classes_per_sample:
            instances = random.sample(instances, self.num_classes_per_sample)

        conversations, masks = self._generate_conversations(item, instances, dataset_name, h, w)

        sam_img = self.transform.apply_image(img)
        sam_img = self.preprocess(torch.from_numpy(sam_img).permute(2, 0, 1))
        vlm_img = Image.fromarray(img)

        if masks:
            mask_tensor = torch.from_numpy(np.stack(masks, 0)).float()
        else:
            mask_tensor = torch.zeros((0, h, w), dtype=torch.float)

        label = torch.ones((h, w), dtype=torch.long) * self.ignore_label

        return img_path, sam_img, vlm_img, conversations, mask_tensor, label, sam_img.shape[-2:], [], []
