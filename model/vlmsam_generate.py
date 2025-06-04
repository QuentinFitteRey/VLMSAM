import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from model.segment_anything import build_sam_vit_h

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
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


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
        self.visual_model = build_sam_vit_h(self.vision_pretrained).to(kwargs['torch_dtype'])
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if kwargs['train_mask_decoder']:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = kwargs['out_dim']
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
            kwargs['model'], torch_dtype=kwargs['torch_dtype'],attn_implementation=kwargs['attention']
        )
        self.model_vlmSamSeg = VlmSamSegModel(config, **kwargs).to(kwargs["torch_dtype"])
        self.processor = AutoProcessor.from_pretrained(kwargs['model'])
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model_vlmSamSeg.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(self,
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
                      **kwargs):
        image_embeddings = self.get_visual_embs(images)  
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1, "Mismatch between images and offset"
        if not inference:
            output = self.vlm.forward(
                input_ids=input_ids,
                pixel_values=vlm_inputs['pixel_values'],
                image_grid_thw=vlm_inputs['image_grid_thw'],
                labels=labels,
                attention_mask=attention_masks,
                use_cache=False,
                output_hidden_states=True,
            )
        else:
            output = self.vlm.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                pixel_values=vlm_inputs['pixel_values'],
                image_grid_thw=vlm_inputs['image_grid_thw'],
                attention_mask=attention_masks,
                return_dict_in_generate=True,
                use_cache=False,
                output_hidden_states=True,
            )
            
        output_hidden_states = output.hidden_states 
        if not inference:
            output_hidden_states = output_hidden_states[-1]
            seg_token_mask = input_ids[:, :] == self.seg_token_idx
        else:
            output_hidden_states = output_hidden_states[-1][-1]
            seg_token_mask = output.sequences[:, 1:] == self.seg_token_idx
        
        hidden_states = []
        assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
        hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_offset.device).long(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model_vlmSamSeg.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": gt_masks}

        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], \
                "gt_mask.shape: {}, pred_mask.shape: {}".format(gt_mask.shape, pred_mask.shape)
            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss = ce_loss + mask_loss

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": gt_masks,
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "mask_bce_loss": mask_bce_loss,
                    "mask_dice_loss": mask_dice_loss,
                    "mask_loss": mask_loss}
        return {"loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss}

    def evaluate(self,
                 inputs_qwen,
                 image_classical,
                 resize_list,
                 original_size_list,
                 max_new_tokens=20,
                 tokenizer=None):
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
            logger.info("final element %s",outputs.hidden_states[-1][-1])
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            
            hidden_states = []
            assert len(self.model_vlmSamSeg.text_hidden_fcs) == 1
            hidden_states.append(self.model_vlmSamSeg.text_hidden_fcs[0](outputs.hidden_states[-1][-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            image_embeddings = self.get_visual_embs(image_classical)
            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.model_vlmSamSeg.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model_vlmSamSeg.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model_vlmSamSeg.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model_vlmSamSeg.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])
        torch.cuda.empty_cache()
        return output_ids, pred_masks