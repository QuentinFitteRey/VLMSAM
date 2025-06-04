import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from PIL import Image

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        model_name="qwen_vl",  
    ):
        """
        Initialize the ReferSegDataset with dataset-specific configurations.

        Args:
            base_image_dir (str): Base directory for dataset files.
            tokenizer: Tokenizer for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): Number of classes to sample per image.
            exclude_val (bool): Whether to exclude validation data.
            refer_seg_data (str): Datasets to use, separated by "||".
            model_name (str): Model name ("llava" or "qwen_vl").
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

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split("||")  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            splitBy = "umd" if ds == "refcocog" else "unc"
            refer_api = G_REFER(DATA_DIR, ds, splitBy) if ds == "grefcoco" else REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {
                "images": [],
                "annotations": refer_api.Anns,
                "img2refs": {}
            }

            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(DATA_DIR, "images/saiapr_tc-12", item["file_name"])
                else:
                    item["file_name"] = os.path.join(DATA_DIR, "images/mscoco/images/train2014", item["file_name"])
                refer_seg_ds["images"].append(item)

            for ref in refs_train:
                image_id = ref["image_id"]
                refer_seg_ds["img2refs"][image_id] = refer_seg_ds["img2refs"].get(image_id, []) + [ref]

            print(f"Dataset {ds} (refs {splitBy}) (train split) has {len(refer_seg_ds['images'])} images and {len(refer_seg_ds['annotations'])} annotations.")
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input for SAM."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        """Fetch a sample with dataset-specific logic."""
        ds = random.choice(self.refer_seg_ds_list)
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if not refs:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"])
                ann_ids.append(ref["ann_id"])

        sampled_inds = random.sample(range(len(sents)), min(self.num_classes_per_sample, len(sents)))
        sampled_sents = [sents[i] for i in sampled_inds]
        sampled_ann_ids = [ann_ids[i] for i in sampled_inds]
        sampled_classes = sampled_sents

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  

        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        questions = [random.choice(self.short_question_list).format(class_name=c.lower()) for c in sampled_classes]
        answers = [random.choice(self.answer_list) for _ in sampled_classes]

        conversations = []
        for q, a in zip(questions, answers):
            conv = get_default_conv_template(self.model_name).copy()
            if "qwen" in self.model_name:
                user_message = f"<|vision_start|><|image_pad|><|vision_end>\n{q}"
            else:
                user_message = "<image>" + "\n" + q
            conv.append_message(conv.roles[0], user_message)
            conv.append_message(conv.roles[1], a)
            conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                for ann_id_i in ann_id:
                    ann = annotations[ann_id_i]
                    if not ann["segmentation"]:
                        m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    else:
                        if isinstance(ann["segmentation"][0], list):  # polygon
                            rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                        else:
                            rle = ann["segmentation"]
                            for i in range(len(rle)):
                                if not isinstance(rle[i]["counts"], bytes):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                        m = mask.decode(rle)
                        m = np.sum(m, axis=2).astype(np.uint8)
                    m_final = m_final | m
                masks.append(m_final)
            else:
                ann = annotations[ann_id]
                if not ann["segmentation"]:
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    if isinstance(ann["segmentation"][0], list):  # polygon
                        rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                    m = np.sum(m, axis=2).astype(np.uint8)
                masks.append(m)

        masks = torch.from_numpy(np.stack(masks, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
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
