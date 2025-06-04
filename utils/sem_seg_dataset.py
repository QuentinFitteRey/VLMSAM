import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from .conversation import get_default_conv_template  
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, MULTI_CLASS_QUESTION_LIST, INST_ANSWER_LIST


# Initialization functions for each dataset

def init_mapillary(base_image_dir):
    """Initialize Mapillary dataset."""
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    """Initialize ADE20K dataset."""
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = [x[:-4] for x in image_ids if x.endswith(".jpg")]
    ade20k_images = [
        os.path.join(base_image_dir, "ade20k", "images", "training", f"{image_id}.jpg")
        for image_id in ade20k_image_ids
    ]
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    """Initialize COCO-Stuff dataset."""
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco")
        for x in cocostuff_labels
    ]
    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    """Initialize PACO-LVIS dataset."""
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    """Initialize Pascal-Part dataset."""
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        sem_seg_p=[1.0, 0.0, 0.0],
        model_name="qwen_vl",
    ):
        """
        Initialize the SemSegDataset with dataset-specific configurations.
        
        Args:
            base_image_dir (str): Base directory for dataset files.
            tokenizer: Tokenizer for text processing.
            samples_per_epoch (int): Number of samples per epoch.
            precision (str): Data precision ("fp32" or "fp16").
            image_size (int): Target image size for resizing.
            num_classes_per_sample (int): Number of classes to sample per image.
            exclude_val (bool): Whether to exclude validation data.
            sem_seg_data (str): Datasets to use, separated by "||".
            sem_seg_p (list of float): Probabilities for sampling 1, 2 or 3 classes.
            model_name (str): Model identifier ("llava" or "qwen_vl").
        """
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.model_name = model_name.lower()
        self.sem_seg_p = sem_seg_p

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.multi_class_question_list = MULTI_CLASS_QUESTION_LIST
        self.inst_answer_list = INST_ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval(f"init_{ds}")(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }
        print("sem_seg_p: ", sem_seg_p)

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
        ds = random.choice(self.sem_seg_datas)

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if not anns:
                return self.__getitem__(0)
            sampled_anns = random.sample(anns, min(self.num_classes_per_sample, len(anns)))
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    name = random.choice([f"{obj} {part}", f"the {part} of the {obj}"])
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image_list, label_list = self.data2list[ds]
            idx = random.randint(0, len(image_list) - 1)
            image_path = image_list[idx]
            label_path = label_list[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = np.array(Image.open(label_path))
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255

            unique_label = [l for l in np.unique(label) if l != 255]
            if not unique_label:
                return self.__getitem__(0)
            classes = [self.data2classes[ds][l] for l in unique_label]
            sampled_classes = random.sample(classes, min(self.num_classes_per_sample, len(classes)))

        questions = []
        answers = []
        class_ids = []
        i = 0
        while i < len(sampled_classes):
            number = np.random.choice([1, 2, 3], p=self.sem_seg_p)
            number = min(len(sampled_classes) - i, number)

            if number == 1:
                sampled_cls = sampled_classes[i]
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=sampled_cls.lower()))
                answers.append(random.choice(self.answer_list))
                if ds not in ["paco_lvis", "pascal_part"]:
                    class_id = self.data2classes[ds].tolist().index(sampled_cls)
                    class_ids.append(class_id)
            else:
                text = "the "
                for idx2, c in enumerate(sampled_classes[i:i + number]):
                    text += c
                    if idx2 < number - 2:
                        text += ", "
                    elif idx2 == number - 2:
                        text += " and " if idx2 == 0 else ", and "
                question_template = random.choice(self.multi_class_question_list)
                questions.append(question_template.format(classes=text.lower()))

                seg_tokens = ""
                for idx2 in range(number):
                    seg_tokens += "[SEG]"
                    if idx2 < number - 2:
                        seg_tokens += ", "
                    elif idx2 == number - 2:
                        seg_tokens += " and " if idx2 == 0 else ", and "
                answer_template = random.choice(self.inst_answer_list)
                answers.append(answer_template.format(seg_tokens=seg_tokens))
                if ds not in ["paco_lvis", "pascal_part"]:
                    for c in sampled_classes[i:i + number]:
                        class_id = self.data2classes[ds].tolist().index(c)
                        class_ids.append(class_id)
            i += number

        conversations = []
        for q, a in zip(questions, answers):
            conv = get_default_conv_template(self.model_name).copy()
            if "qwen" in self.model_name:
                user_message = f"<|vision_start|><|image_pad|><|vision_end|>\n{q}"
            else:
                user_message = "<image>" + "\n" + q
            conv.append_message(conv.roles[0], user_message)
            conv.append_message(conv.roles[1], a)
            conversations.append(conv.get_prompt())

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_sam = self.preprocess(image_tensor)
        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)  
        else:
            image_vlm = image_sam

        resize = image.shape[:2]

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label_map = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            label_map = torch.from_numpy(label).long()
            masks = torch.stack([label_map == cid for cid in class_ids], dim=0)
        return (
            image_path,
            image_sam,
            image_vlm,
            conversations,
            masks,
            label_map,
            resize,
            questions,
            sampled_classes,
        )
