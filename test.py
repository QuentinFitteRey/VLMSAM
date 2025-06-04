import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from peft import LoraConfig, get_peft_model

from model.vlmsam import VlmSamSegForCausalLM
from utils.dataset import ValDataset, collate_fn
from utils.utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda2, intersectionAndUnionGPU

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(args):
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=32768, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--workers", default=15, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--vision_pretrained", default="./weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--resume", required=True, type=str)
    return parser.parse_args(args)

def setup_ddp(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size == 1:
        use_ddp = False
        device = torch.device('cuda:0')
    else:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        use_ddp = True
        device = torch.device(f'cuda:{rank}')
    return device, world_size, rank, use_ddp

def validate(val_loader, model, global_step, args, rank, device, use_ddp):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model.eval()
    with torch.no_grad():
        for input_dict in val_loader:
            input_dict = dict_to_cuda2(input_dict)
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
            with torch.autocast(device_type='cuda', dtype=dtype):
                output_dict = model(**input_dict)
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()

            intersection, union, acc_iou_sum = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                iou_i = intersection_i / (union_i + 1e-5)
                iou_i[union_i == 0] = 1.0
                acc_iou_sum += iou_i

            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            acc_iou_meter.update(acc_iou_sum.cpu().numpy(), n=masks_list.shape[0])

    if use_ddp:
        intersection_sum = torch.tensor(intersection_meter.sum, device=device)
        union_sum = torch.tensor(union_meter.sum, device=device)
        dist.all_reduce(intersection_sum)
        dist.all_reduce(union_sum)
        intersection_meter.sum = intersection_sum.cpu().numpy()
        union_meter.sum = union_sum.cpu().numpy()

        acc_iou_sum_total = torch.tensor(acc_iou_meter.sum, device=device)
        acc_iou_count_total = torch.tensor(acc_iou_meter.count, device=device)
        dist.all_reduce(acc_iou_sum_total)
        dist.all_reduce(acc_iou_count_total)
    else:
        acc_iou_sum_total = acc_iou_meter.sum
        acc_iou_count_total = acc_iou_meter.count

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = (acc_iou_sum_total / acc_iou_count_total)[1].item() if acc_iou_count_total > 0 else 0

    if rank == 0:
        print(f"Step: {global_step}, giou: {giou:.4f}, ciou: {ciou:.4f}")

    return giou, ciou

def main(args):
    device, world_size, rank, use_ddp = setup_ddp(args)

    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    tokenizer = processor.tokenizer
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "vision_pretrained": args.vision_pretrained,
        "model": args.version,
        "out_dim": args.out_dim,
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": args.seg_token_idx,
        "use_mm_start_end": args.use_mm_start_end,
        "torch_dtype": torch_dtype,
        "attention": args.attention
    }

    config = transformers.AutoConfig.from_pretrained(args.version)
    model = VlmSamSegForCausalLM(config, **model_args).to(device)
    model.processor = processor

    model.vlm.config.eos_token_id = tokenizer.eos_token_id
    model.vlm.config.bos_token_id = tokenizer.bos_token_id
    model.vlm.config.pad_token_id = tokenizer.pad_token_id
    model.vlm.enable_input_require_grads()
    for p in model.vlm.visual.parameters():
        p.requires_grad = False

    if args.lora_r > 0:
        lora_target_modules = args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.vlm = get_peft_model(model.vlm, lora_config)
        model.vlm.print_trainable_parameters()

    model.vlm.resize_token_embeddings(len(tokenizer))

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    val_dataset = ValDataset(args.dataset_dir, tokenizer, args.val_dataset, args.image_size)
    if use_ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=lambda batch: collate_fn(
                batch,
                tokenizer=tokenizer,
                processor=processor,
                model_name="qwen_vl",
                local_rank=rank,
            ),
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(
                batch,
                tokenizer=tokenizer,
                processor=processor,
                model_name="qwen_vl",
            ),
        )
    print(f"Validating with {len(val_dataset)} examples.")

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found at {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not use_ddp and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    giou, ciou = validate(val_loader, model, global_step=0, args=args, rank=rank, device=device, use_ddp=use_ddp)
    if rank == 0:
        print(f"Evaluation results: giou={giou:.4f}, ciou={ciou:.4f}")

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
