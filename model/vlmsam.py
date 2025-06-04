import logging
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from model.segment_anything import build_sam_vit_h
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Initializing model...")


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class VlmSamSegModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(VlmSamSegModel, self).__init__()
        self.config = config

        if not kwargs["train_mask_decoder"]:
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_vlmSamSeg_modules(self.config, kwargs)

    def initialize_vlmSamSeg_modules(self, config, kwargs):
        # Build SAM visual model
        self.visual_model = build_sam_vit_h(self.vision_pretrained).to(kwargs["torch_dtype"])
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if kwargs["train_mask_decoder"]:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer for text embeddings
        in_dim = config.hidden_size
        out_dim = kwargs["out_dim"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class VlmSamSegForCausalLM(nn.Module):
    def __init__(self, config, **kwargs):
        super(VlmSamSegForCausalLM, self).__init__()
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            kwargs["model"],
            torch_dtype=kwargs["torch_dtype"],
            attn_implementation=kwargs["attention"]
        )
        self.model_vlmSamSeg = VlmSamSegModel(config, **kwargs).to(kwargs["torch_dtype"])
        self.processor = AutoProcessor.from_pretrained(kwargs["model"])
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    @staticmethod
    def adjust_indices_order(pred_indices: np.ndarray, gt_indices: np.ndarray):
        adjusted_gt_indices = np.empty_like(gt_indices)
        sorted_pred_indices = np.argsort(pred_indices)
        for i, sorted_idx in enumerate(sorted_pred_indices):
            adjusted_gt_indices[i] = gt_indices[sorted_idx]
        return np.arange(len(pred_indices)), adjusted_gt_indices

    def hungarian_matcher(self, pred_masks: List[torch.Tensor], gt_masks: List[torch.Tensor]):
        pred_masks = torch.stack([m.squeeze(0) for m in pred_masks]).flatten(1)
        gt_masks = torch.stack([m.squeeze(0) for m in gt_masks]).flatten(1)
        dice_loss_cur = batch_dice_loss(pred_masks, gt_masks)
        sigmoid_ce_loss_cur = batch_sigmoid_ce_loss(pred_masks, gt_masks)
        cost_matrix = dice_loss_cur + sigmoid_ce_loss_cur
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.detach().cpu())
        adjust_pred_indices, adjust_gt_indices = self.adjust_indices_order(pred_indices, gt_indices)
        return adjust_pred_indices, adjust_gt_indices

    def hungarian_matcher_batch(self, pred_masks: List[List[torch.Tensor]], gt_masks: List[List[torch.Tensor]], change_list: List[List[int]]):
        reordered_gt_masks = []
        for batch_idx, groups in enumerate(change_list):
            batch_pred_masks = pred_masks[batch_idx]
            batch_gt_masks = gt_masks[batch_idx]
            reordered_batch_gt_masks = batch_gt_masks.clone()
            for group in groups:
                group_pred_masks = batch_pred_masks[group].unsqueeze(1).flatten(1)
                group_gt_masks = batch_gt_masks[group].unsqueeze(1).flatten(1)
                _, group_gt_indices = self.hungarian_matcher(group_pred_masks, group_gt_masks)
                for idx, gt_idx in enumerate(group_gt_indices):
                    reordered_batch_gt_masks[group[idx]] = batch_gt_masks[group[gt_idx]]
            reordered_gt_masks.append(reordered_batch_gt_masks)
        return reordered_gt_masks

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                emb = self.model_vlmSamSeg.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
                image_embeddings_list.append(emb)
            torch.cuda.empty_cache()
            return torch.cat(image_embeddings_list, 0)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        vlm_inputs: dict,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        change_list: List[torch.Tensor] = [],
        **kwargs
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"

        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=vlm_inputs["pixel_values"],
            image_grid_thw=vlm_inputs["image_grid_thw"],
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states[-1]
        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = []
        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_offset.device).long(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_list

        pred_masks = []
        for i, text_embeds in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=text_embeds.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)
            low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list
        for idx in range(len(change_list)):
            if isinstance(change_list[idx], list):
                gt_masks_cur = self.hungarian_matcher_batch(
                    [pred_masks[idx]], [gt_masks[idx]], [change_list[idx]]
                )
                gt_masks[idx] = gt_masks_cur[0]

        logits = output.logits
        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] != pred_mask.shape[0]:
                num_masks += 1
                continue

            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        inputs_qwen: dict,
        image_classical: torch.FloatTensor,
        resize_list: List[tuple],
        original_size_list: List[tuple],
        max_new_tokens: int = 128,
        tokenizer=None
    ):
        with torch.no_grad():
            outputs = self.vlm.generate(
                **inputs_qwen,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            output_ids = outputs.sequences
            decoded = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            logger.info("Decoded Output: %s", decoded)

            output_hidden_states = outputs.hidden_states[-1][-1]
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            hidden_states = []
            assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
            hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

            pred_embeddings_list = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_list.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_list

            image_embeddings = self.get_visual_embs(image_classical)
            pred_masks = []
            for i, text_embeds in enumerate(pred_embeddings):
                sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=text_embeds.unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(text_embeds.dtype)
                low_res_masks, _ = self.model_vlmSamSeg.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        torch.cuda.empty_cache()
        return output_ids, pred_masks
