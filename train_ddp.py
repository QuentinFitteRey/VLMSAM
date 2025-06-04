import argparse
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import wandb

from model.vlmsam import VlmSamSegForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

# Set TOKENIZERS_PARALLELISM to false to avoid tokenizer issues with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args):
    parser = argparse.ArgumentParser(description="VlmSamSeg Model Training with DDP and Resumption")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=32768, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--min_pixels", default=256 * 28 * 28, type=int)
    parser.add_argument("--max_pixels", default=1280 * 28 * 28, type=int)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dataset", default="sem_seg||refer_seg||vqa||reason_seg||cot", type=str)
    parser.add_argument("--sample_rates", default="8,3,3,1,3", type=str)
    parser.add_argument("--sem_seg_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--cot_data", default="caption||cot||conversation", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./testaaaa", type=str)
    parser.add_argument("--exp_name", default="vlmsamseg", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=5000, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.00003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="./weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch (overridden if resuming)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--val_freq", default=50, type=int, help="Validation frequency in optimization steps")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators")
    return parser.parse_args(args)

def setup_ddp():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return torch.device(f'cuda:{rank}'), world_size, rank

def save_checkpoint(model, optimizer, scheduler, epoch, best_score, cur_ciou, args, rank):
    if rank == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "cur_ciou": cur_ciou,
        }
        checkpoint_path = os.path.join(args.log_base_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    dist.barrier()  # Synchronize all processes after saving

# Validation Function
def validate(val_loader, model, global_optim_step, writer, args, rank, device):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    val_loss_meter = AverageMeter("ValLoss", ":.4f")
    ce_loss_meter = AverageMeter("CeLoss", ":.4f")
    mask_bce_loss_meter = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_loss_meter = AverageMeter("MaskDICELoss", ":.4f")
    mask_loss_meter = AverageMeter("MaskLoss", ":.4f")

    model.eval()
    with torch.no_grad():
        for input_dict in val_loader:
            input_dict = dict_to_cuda(input_dict)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32):
                output_dict = model(**input_dict)
                loss = output_dict["loss"].item()
            val_loss_meter.update(loss, input_dict["images"].size(0))
            ce_loss_meter.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
            mask_bce_loss_meter.update(output_dict["mask_bce_loss"].item(), input_dict["images"].size(0))
            mask_dice_loss_meter.update(output_dict["mask_dice_loss"].item(), input_dict["images"].size(0))
            mask_loss_meter.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

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

    # Loss meters
    val_loss_sum = torch.tensor(val_loss_meter.sum, device=device)
    val_loss_count = torch.tensor(val_loss_meter.count, device=device)
    dist.all_reduce(val_loss_sum)
    dist.all_reduce(val_loss_count)
    val_loss_avg = val_loss_sum / val_loss_count if val_loss_count > 0 else 0

    ce_loss_sum = torch.tensor(ce_loss_meter.sum, device=device)
    ce_loss_count = torch.tensor(ce_loss_meter.count, device=device)
    dist.all_reduce(ce_loss_sum)
    dist.all_reduce(ce_loss_count)
    ce_loss_avg = ce_loss_sum / ce_loss_count if ce_loss_count > 0 else 0

    mask_bce_loss_sum = torch.tensor(mask_bce_loss_meter.sum, device=device)
    mask_bce_loss_count = torch.tensor(mask_bce_loss_meter.count, device=device)
    dist.all_reduce(mask_bce_loss_sum)
    dist.all_reduce(mask_bce_loss_count)
    mask_bce_loss_avg = mask_bce_loss_sum / mask_bce_loss_count if mask_bce_loss_count > 0 else 0

    mask_dice_loss_sum = torch.tensor(mask_dice_loss_meter.sum, device=device)
    mask_dice_loss_count = torch.tensor(mask_dice_loss_meter.count, device=device)
    dist.all_reduce(mask_dice_loss_sum)
    dist.all_reduce(mask_dice_loss_count)
    mask_dice_loss_avg = mask_dice_loss_sum / mask_dice_loss_count if mask_dice_loss_count > 0 else 0

    mask_loss_sum = torch.tensor(mask_loss_meter.sum, device=device)
    mask_loss_count = torch.tensor(mask_loss_meter.count, device=device)
    dist.all_reduce(mask_loss_sum)
    dist.all_reduce(mask_loss_count)
    mask_loss_avg = mask_loss_sum / mask_loss_count if mask_loss_count > 0 else 0

    # Compute final metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = (acc_iou_sum_total / acc_iou_count_total)[1].item() if acc_iou_count_total > 0 else 0

    if rank == 0:
        print(f"global_optim_step: {global_optim_step}, giou: {giou:.4f}, ciou: {ciou:.4f}, val_loss: {val_loss_avg:.4f}")
        wandb.log({
            "val/loss": val_loss_avg,
            "val/ce_loss": ce_loss_avg,
            "val/mask_bce_loss": mask_bce_loss_avg,
            "val/mask_dice_loss": mask_dice_loss_avg,
            "val/mask_loss": mask_loss_avg,
            "val/giou": giou,
            "val/ciou": ciou,
            "global_optim_step": global_optim_step
        })
        if writer:
            writer.add_scalar("val/loss", val_loss_avg, global_optim_step)
            writer.add_scalar("val/giou", giou, global_optim_step)
            writer.add_scalar("val/ciou", ciou, global_optim_step)

    return giou, ciou

def train(train_loader, val_loader, model, optimizer, scheduler, epoch, writer, args, rank, best_score, cur_ciou, device):
    optim_steps_per_epoch = args.steps_per_epoch // args.grad_accumulation_steps
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        optim_steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, mask_bce_losses, mask_dice_losses],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    end = time.time()
    optim_step = 0
    for step, input_dict in enumerate(train_loader):
        if step >= args.steps_per_epoch:
            break
        input_dict = dict_to_cuda(input_dict)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32):
            output_dict = model(**input_dict)
            loss = output_dict["loss"] / args.grad_accumulation_steps
        loss.backward()

        if (step + 1) % args.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            optim_step += 1
            global_optim_step = epoch * optim_steps_per_epoch + optim_step

            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(output_dict["loss"].item(), input_dict["images"].size(0))
            ce_losses.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
            mask_bce_losses.update(output_dict["mask_bce_loss"].item(), input_dict["images"].size(0))
            mask_dice_losses.update(output_dict["mask_dice_loss"].item(), input_dict["images"].size(0))
            mask_losses.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

            if rank == 0 and global_optim_step % args.print_freq == 0:
                progress.display(global_optim_step)
                wandb.log({
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/mask_bce_loss": mask_bce_losses.avg,
                    "train/mask_dice_loss": mask_dice_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_optim_step": global_optim_step
                })
                if writer:
                    writer.add_scalar("train/loss", losses.avg, global_optim_step)

            if val_loader is not None and global_optim_step % args.val_freq == 0:
                giou, ciou = validate(val_loader, model, global_optim_step, writer, args, rank, device)
                if rank == 0 and giou > best_score[0]:
                    best_score[0] = giou
                    cur_ciou[0] = ciou
                    torch.save(model.state_dict(), os.path.join(args.log_base_dir, f"best_model_epoch{epoch}_step{global_optim_step}_giou{giou:.3f}_ciou{ciou:.3f}.pth"))

def main(args):
    device, world_size, rank = setup_ddp()

    if rank == 0:
        wandb.init(project="vlmsamseg_training", config=vars(args))
        os.makedirs(args.log_base_dir, exist_ok=True)
        writer = SummaryWriter(args.log_base_dir)
    else:
        writer = None
    dist.barrier()  

    processor = transformers.AutoProcessor.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    tokenizer = processor.tokenizer
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "model": args.version,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
        "torch_dtype": torch_dtype,
        "attention": args.attention
    }
    config = transformers.AutoConfig.from_pretrained(args.version)
    model = VlmSamSegForCausalLM(config, **model_args).to(device)

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

    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs"]) or \
           "mask_decoder" in n:
            p.requires_grad = True
    model.vlm.resize_token_embeddings(len(tokenizer))

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=0.0
    )
    optim_steps_per_epoch = args.steps_per_epoch // args.grad_accumulation_steps
    total_optim_steps = args.epochs * optim_steps_per_epoch
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_optim_steps
    )

    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        cot_data=args.cot_data,
        explanatory=args.explanatory,
        sem_seg_p=[0.40, 0.3, 0.3]
        )
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            processor=processor,
            local_rank=rank,
        ),
    )

    if not args.no_eval or args.eval_only:
        val_dataset = ValDataset(args.dataset_dir, tokenizer, args.val_dataset, args.image_size)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                processor=processor,
                model_name="qwen_vl",
                local_rank=rank,
            ),
        )
        print(f"Training with {len(train_dataset)} examples, validating with {len(val_dataset)}.")
    else:
        val_loader = None
        print(f"Training with {len(train_dataset)} examples.")

    best_score = [0.0]
    cur_ciou = [0.0]

    if args.resume and not args.eval_only:  
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            args.start_epoch = checkpoint["epoch"] + 1  
            best_score[0] = checkpoint["best_score"]
            cur_ciou[0] = checkpoint["cur_ciou"]
            print(f"Resumed from epoch {checkpoint['epoch']}, best_score: {best_score[0]}, cur_ciou: {cur_ciou[0]}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch.")
    elif not args.eval_only:
        print("No resume checkpoint specified, starting from scratch.")

    if args.eval_only:
        if not args.resume:
            raise ValueError("For eval-only, please specify a checkpoint to load using --resume")
        checkpoint_path = args.resume
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)  
        model.load_state_dict(state_dict)
        model.eval()
        if val_loader is not None:
            giou, ciou = validate(val_loader, model, 0, writer, args, rank, device)
            if rank == 0:
                print(f"Evaluation results: giou={giou:.4f}, ciou={ciou:.4f}")
        else:
            print("No validation dataset specified.")
        if rank == 0:
            if writer:
                writer.close()
            wandb.finish()
        dist.destroy_process_group()
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, val_loader, model, optimizer, scheduler, epoch, writer, args, rank, best_score, cur_ciou, device)
        save_checkpoint(model, optimizer, scheduler, epoch, best_score[0], cur_ciou[0], args, rank)

    if rank == 0:
        if writer:
            writer.close()
        wandb.finish()
    dist.destroy_process_group()
    
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)