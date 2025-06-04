#!/usr/bin/env python
import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig
from PIL import Image
import requests

from qwen_vl_utils import process_vision_info
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from model.vlmsam import VlmSamSegForCausalLM 
import types


def parse_args(args):
    parser = argparse.ArgumentParser(description="VlmSamSeg Chat with Qwen2.5-VL & SAM")
    parser.add_argument(
        "--version", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Your model version"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="Precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="Image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--resume", required=True, type=str, help="Path to checkpoint to load"
    )
    return parser.parse_args(args)


def prepare_qwen_vl_inputs(
    self,
    batch_messages: List[List[dict]],
    batch_image_sources: List[str],
    batch_vision_infos: List[dict],
):
    """
    Prepare batched inputs for Qwen2.5-VL generation.
    Each element in batch_messages should be a list of message dicts in Qwen’s expected format.
    """
    images, texts = [], []
    for messages, image_source in zip(batch_messages, batch_image_sources):
        if image_source.startswith("http"):
            image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")
        images.append(image)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
    return self.processor(text=texts, images=images, vision_infos=batch_vision_infos, return_tensors="pt")


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    return F.pad(x, (0, padw, 0, padh))


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    print("Initializing...")

    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    tokenizer = processor.tokenizer

    # Ensure the tokenizer has the segmentation token.
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,
        "model": args.version,
        "attention": "flash_attention_2",
        "train_mask_decoder": True,
        "out_dim": 256,
    }

    config = AutoConfig.from_pretrained(args.version)
    config.train_mask_decoder = False
    config.out_dim = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VlmSamSegForCausalLM(config, seg_token_idx=args.seg_token_idx, **kwargs).to(device)

    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id
    model.vlm.resize_token_embeddings(len(tokenizer))

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found at {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    model.prepare_qwen_vl_inputs = types.MethodType(prepare_qwen_vl_inputs, model)
    transform = ResizeLongestSide(args.image_size)

    while True:
        prompt = input("Please input your prompt (or press Enter to exit): ").strip()
        if prompt == "":
            break

        image_path = input("Please input the image path or URL: ").strip()
        if not os.path.exists(image_path) and not image_path.startswith("http"):
            print("File not found:", image_path)
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs_qwen = (
            processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            .to(device)
        )

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        image_classical = transform.apply_image(image_np)
        resize_list = [image_classical.shape[:2]]
        image_classical = (
            preprocess(
                torch.from_numpy(image_classical).permute(2, 0, 1).contiguous(),
                img_size=args.image_size,
            )
            .unsqueeze(0)
            .to(device)
            .to(torch_dtype)
        )

        output_ids, pred_masks = model.evaluate(
            inputs_qwen,
            image_classical,
            resize_list,
            original_size_list,
            max_new_tokens=128,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        seg_text_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        seg_text_output = seg_text_output.replace("\n", "").replace("  ", " ")
        print("\n--- Segmentation Branch Output ---")
        print(seg_text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue
            pred_mask_np = pred_mask.detach().cpu().numpy()[0] > 0

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(args.vis_save_path, f"{base_name}_mask_{i}.jpg")
            cv2.imwrite(mask_path, pred_mask_np.astype(np.uint8) * 255)
            print(f"{mask_path} saved.")

            masked_img = image_np.copy()
            masked_img[pred_mask_np] = (
                image_np * 0.5 + pred_mask_np[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask_np]
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
            masked_path = os.path.join(args.vis_save_path, f"{base_name}_masked_img_{i}.jpg")
            cv2.imwrite(masked_path, masked_img)
            print(f"{masked_path} saved.")


if __name__ == "__main__":
    main(sys.argv[1:])
