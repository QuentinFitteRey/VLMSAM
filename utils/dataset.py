import glob
import os
import random

import cv2
import numpy as np
import torch
import re
import json
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .cot_dataset import COTDataset
from .utils import DEFAULT_SEMATIC_SEG, DEFAULT_INSTANT_SEG, DEFAULT_IMAGE_TOKEN
from .vqa_dataset import VQADataset

from typing import List, Union

from transformers import ProcessorMixin


def collate_fn(batch, tokenizer=None, processor=None, model_name="qwen_vl", debug=False, use_mm_start_end=True, local_rank=-1):
    """
    Custom collate function for Qwen-VL model to process images once and handle multiple conversations efficiently.

    Args:
        batch: List of tuples containing (image_path, images, image_vlm, conversations, masks, label, resize,
                questions, sampled_classes, inference)
        processor: Instance of Qwen2_5_VLProcessor or similar
        tokenizer: Optional tokenizer (defaults to processor.tokenizer)
        model_name: Name of the model (e.g., "qwen_vl")
        use_mm_start_end: Boolean to use multimodal start/end tokens
        local_rank: Local rank for distributed training (default: -1)

    Returns:
        Dictionary with batched inputs matching the expected format.
    """
    tokenizer = tokenizer or processor.tokenizer
    model_name = model_name.lower()

    image_path_list = []
    images_list = []
    image_vlm_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    change_all_list = []
    semantic_all_lst = []

    for (
        image_path,
        images,
        image_vlm,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        if isinstance(image_path, list):
            change_onelst = image_path[1]
            change_all_list.append(change_onelst)
            image_path = image_path[0]
        else:
            change_all_list.append(-1)
        if isinstance(images, list):
            semantic_all_lst.append(images[1])
            images = images[0]
        else:
            semantic_all_lst.append([-1] * masks.shape[0])
        image_path_list.append(image_path)
        images_list.append(images)
        image_vlm_list.append(image_vlm)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if "qwen" in model_name:
        image_inputs = processor.image_processor(image_vlm_list, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        nb_token_list = []
        for i in range(len(image_grid_thw)):
            sample_image_grid_thw = image_grid_thw[i]
            nb_tokens = sample_image_grid_thw.prod()
            nb_token_list.append(nb_tokens)

        all_modified_conversations = []
        conversation_to_sample_idx = []
        duplicated_image_grid_thw = []
        for i in range(len(offset_list) - 1):
            start = offset_list[i]
            end = offset_list[i + 1]
            sample_conversations = conversation_list[start:end]

            sample_image_grid_thw = image_grid_thw[i]
            merge_length = processor.image_processor.merge_size ** 2
            nb_tokens = nb_token_list[i]
            num_placeholders = nb_tokens // merge_length

            for conv in sample_conversations:
                if processor.image_token in conv:
                    modified_conv = conv.replace(
                        processor.image_token,
                        "<|placeholder|>" * num_placeholders,
                        1
                    )
                    modified_conv = modified_conv.replace("<|placeholder|>", processor.image_token)
                else:
                    modified_conv = conv
                all_modified_conversations.append(modified_conv)
                conversation_to_sample_idx.append(i)

            num_conversations = offset_list[i + 1] - offset_list[i]
            duplicated_image_grid_thw.extend([sample_image_grid_thw] * num_conversations)

        text_inputs = tokenizer(all_modified_conversations, return_tensors="pt", padding=True)
        input_ids = text_inputs["input_ids"]
        attention_masks = text_inputs["attention_mask"]
        semantic_ids_lst = []
        for sublist in semantic_all_lst:
            ids_sublist = []
            if -1 not in sublist:
                for item in sublist:
                    token_ids = tokenizer(item).input_ids
                    if len(token_ids) != 2:
                        ids_sublist.append(torch.tensor(-100, dtype=torch.long))
                    else:
                        ids_sublist.append(torch.tensor(token_ids[-1], dtype=torch.long))
                semantic_ids_lst.append(ids_sublist)
            else:
                semantic_ids_lst.append([torch.tensor(-100, dtype=torch.long)] * len(sublist))

        targets = input_ids.clone()
        targets[targets == tokenizer.pad_token_id] = -100
        assistant_start_str = "<|im_start|>assistant\n"
        assistant_end_str = "<|im_end|>"

        assistant_start_tokens = tokenizer.encode(assistant_start_str, add_special_tokens=False)
        assistant_end_tokens = tokenizer.encode(assistant_end_str, add_special_tokens=False)

        for idx in range(targets.shape[0]):
            target = targets[idx]

            start_positions = []
            for pos in range(target.size(0) - len(assistant_start_tokens) + 1):
                if target[pos: pos + len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                    start_positions.append(pos + len(assistant_start_tokens))
            if not start_positions:
                raise ValueError(f"Assistant start tokens not found in sample {idx}")

            label = torch.full_like(target, -100)
            for i, start_idx in enumerate(start_positions):
                if i < len(start_positions) - 1:
                    search_end = start_positions[i + 1] - len(assistant_start_tokens)
                else:
                    search_end = target.size(0)
                found_end = False
                for pos in range(start_idx, search_end + 1):
                    if target[pos: pos + len(assistant_end_tokens)].tolist() == assistant_end_tokens:
                        end_idx = pos + 1
                        found_end = True
                        break
                if not found_end:
                    end_idx = search_end
                label[start_idx:end_idx] = target[start_idx:end_idx]
            targets[idx] = label

        cumsum_tokens = torch.cumsum(torch.tensor(nb_token_list), dim=0)
        duplicated_pixel_values = []
        for i in range(len(image_grid_thw)):
            start = cumsum_tokens[i - 1] if i > 0 else 0
            end = cumsum_tokens[i]
            image_segment = pixel_values[start:end]
            num_conversations = offset_list[i + 1] - offset_list[i]
            repeated_segment = image_segment.repeat(num_conversations, 1)
            duplicated_pixel_values.append(repeated_segment)

        pixel_values_batch = torch.cat(duplicated_pixel_values, dim=0)
        duplicated_image_grid_thw = torch.stack(duplicated_image_grid_thw)
        vlm_inputs = {
            "pixel_values": pixel_values_batch,
            "image_grid_thw": duplicated_image_grid_thw
        }

    else:
        raise NotImplementedError(f"Model {model_name} is not supported in this collate_fn")

    if debug:
        return {
            "image_paths": image_path_list,
            "images": torch.stack(images_list, dim=0),
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            "vlm_inputs": vlm_inputs,
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "sampled_classes_list": sampled_classes_list,
            "inference": inferences[0] if inferences else None,
            "conversation_list": all_modified_conversations,
            "change_list": change_all_list,
            "semantic_ids_list": semantic_ids_lst,
        }
    else:
        return {
            "images": torch.stack(images_list, dim=0),
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            "vlm_inputs": vlm_inputs,
            "offset": torch.LongTensor(offset_list),
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "inference": inferences[0] if inferences else None,
            "change_list": change_all_list,
            "semantic_ids_list": semantic_ids_lst,
        }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg||cot",
        sample_rate=[9, 3, 3, 1, 2],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        cot_data="caption||conversation||cot||instance_seg",
        explanatory=0.1,
        sem_seg_p=[1.0, 0.0, 0.0],
        model_name="qwen_vl",
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.model_name = model_name.lower()

        self.datasets = dataset.split("||")
        self.all_datasets = []
        for ds in self.datasets:
            if ds == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        sem_seg_p,
                        model_name=self.model_name,
                    )
                )
            elif ds == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        model_name=self.model_name,
                    )
                )
            elif ds == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        model_name=self.model_name,
                    )
                )
            elif ds == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        model_name=self.model_name,
                    )
                )
            elif ds == "cot":
                self.all_datasets.append(
                    COTDataset(
                        base_image_dir,
                        tokenizer,
                        cot_data,
                        samples_per_epoch,
                        image_size,
                        num_classes_per_sample,
                        model_name=self.model_name
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        val_dataset,
        image_size=1024,
        model_name="qwen_vl"
    ):
        self.base_image_dir = base_image_dir
        self.model_name = model_name.lower()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)

        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            if ds.lower() == "reasoninstanceseg":
                self.data_type = "reason_instance_seg"
                self.specific_dir = os.path.join(self.base_image_dir, "proceed_data")
                self.image_dir = os.path.join(self.base_image_dir, "val_images")
                self.file_count = sum(1 for filename in os.listdir(self.specific_dir) if filename.endswith(".json"))
            else:
                self.data_type = "reason_seg"
                images = sorted(glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")))
                self.images = images
        elif len(splits) == 3:
            ds, splitBy, split = splits
            self.data_type = "refer_seg"
            refer_api = REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {"images": []}
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds.lower() == "refclef":
                    item["file_name"] = os.path.join(self.base_image_dir, "refer_seg", "images/saiapr_tc-12", item["file_name"])
                elif ds.lower() in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(self.base_image_dir, "refer_seg", "images/mscoco/images/train2014", item["file_name"])
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns
            img2refs = {ref["image_id"]: img2refs.get(ref["image_id"], []) + [ref] for ref in refs_val}
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds

        self.ds = splits[0] if splits else None

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        elif self.data_type == "reason_instance_seg":
            return self.file_count
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_multi_mask_from_json(self, item, img):
        height, width = img.shape[:2]
        inform = item["ID"]
        comments = item["English Question"]
        masks = []
        for id in inform:
            mask_np = np.zeros((height, width), dtype=np.uint8)
            points = item["points"][id]["points"]
            label_value = 1
            cv2.polylines(mask_np, np.array([points], dtype=np.int32), True, label_value, 1)
            cv2.fillPoly(mask_np, np.array([points], dtype=np.int32), label_value)
            masks.append(mask_np)
        return masks, comments, True

    def __getitem__(self, idx):
        change_lst_num = None
        masks_json = None
        sampled_sents = None
        is_sentence = False
        ann_ids = None

        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            image_info = refer_seg_ds["images"][idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]
            refs = refer_seg_ds["img2refs"].get(image_id, [])
            if not refs:
                raise ValueError(f"Image {image_id} has no refs")
            sents = [sent["sent"].strip().lower() for ref in refs for sent in ref["sentences"]]
            ann_ids = [ref["ann_id"] for ref in refs for _ in ref["sentences"]]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
            sampled_sents = sents

        elif self.data_type == "reason_instance_seg":
            json_item = os.path.join(self.specific_dir, f"{idx+1}.json")
            with open(json_item, "r") as f:
                item = json.load(f)
            image_path = os.path.join(self.image_dir, item["img_path"])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks_json, sents, is_sentence = self.get_multi_mask_from_json(item, image)
            sampled_sents = [sents]
            change_lst_num = [len(masks_json)]

        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = get_default_conv_template(self.model_name).copy()
        if "qwen" in self.model_name:
            image_token = "<|vision_start|><|image_pad|><|vision_end>"
        else:
            image_token = DEFAULT_IMAGE_TOKEN

        if self.data_type == "reason_instance_seg":
            seg_token = DEFAULT_INSTANT_SEG
        elif self.data_type == "reason_seg":
            seg_token = DEFAULT_SEMATIC_SEG
        else:
            seg_token = ""

        if self.data_type == "reason_instance_seg":
            text = sampled_sents[0].strip()
            conv.messages = []
            conv.append_message(
                conv.roles[0],
                f"{image_token}\n{text} Please output segmentation mask.{seg_token}"
            )
            num_masks = len(masks_json)
            seg_tokens = ""
            for i in range(num_masks):
                seg_tokens += "[SEG]"
                if i < num_masks - 2:
                    seg_tokens += ", "
                elif i == num_masks - 2:
                    seg_tokens += " and " if i == 0 else ", and "
            conv.append_message(conv.roles[1], f"{seg_tokens}.")
            conversations.append(conv.get_prompt())
        else:
            for text in sampled_sents:
                conv.messages = []
                text = text.strip()
                if is_sentence:
                    conv.append_message(
                        conv.roles[0],
                        f"{image_token}\n{text} Please output segmentation mask.{seg_token}"
                    )
                else:
                    conv.append_message(
                        conv.roles[0],
                        f"{image_token}\nWhat is {text} in this image? Please output segmentation mask.{seg_token}"
                    )
                conv.append_message(conv.roles[1], "[SEG]")
                conversations.append(conv.get_prompt())

        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        if "qwen" in self.model_name:
            image_vlm = Image.fromarray(image)
        else:
            clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            image_vlm = clip_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        if self.data_type == "refer_seg":
            masks_list = []
            for i, ann_id in enumerate(ann_ids):
                ann = refer_seg_ds["annotations"][ann_id]
                if not ann["segmentation"] and sampled_sents[i]:
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if isinstance(ann["segmentation"][0], list):
                        rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
                    else:
                        rle = ann["segmentation"]
                        for j in range(len(rle)):
                            if not isinstance(rle[j]["counts"], bytes):
                                rle[j]["counts"] = rle[j]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(m, axis=2).astype(np.uint8)
                masks_list.append(m)
            masks = masks_list
        elif self.data_type == "reason_instance_seg":
            masks = masks_json
        else:
            masks = [mask_json]

        masks = torch.from_numpy(np.stack(masks, axis=0))
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        if self.data_type == "reason_instance_seg":
            change_lst = [[i for i in range(change_lst_num[0])]]
            return (
                [image_path, change_lst],
                image_sam,
                image_vlm,
                conversations,
                masks,
                labels,
                resize,
                None,
                None,
                inference,
            )
        return (
            image_path,
            image_sam,
            image_vlm,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
