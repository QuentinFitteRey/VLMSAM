import argparse
import os
import torch
import transformers
from model.vlmsam import VlmSamSegForCausalLM
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a LoRA-adapted checkpoint, merge the adapters into the base model, and save the merged checkpoint."
    )
    parser.add_argument("--version", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str,
                        help="Pretrained model version")
    parser.add_argument("--resume", required=True, type=str,
                        help="Path to the checkpoint to load")
    parser.add_argument("--save_path", required=True, type=str,
                        help="Path to save the merged model state dict")
    parser.add_argument("--lora_r", default=16, type=int,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", default=32, type=int,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", default=0.05, type=float,
                        help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str,
                        help="Comma separated list of target modules (e.g. 'q_proj,v_proj')")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"],
                        help="Precision for model initialization")
    return parser.parse_args()

def remove_module_prefix(state_dict):
    """
    Remove the "module." prefix from the keys in a state_dict.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    args = parse_args()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    config = transformers.AutoConfig.from_pretrained(args.version)

    model_args = {
        "train_mask_decoder": True,  
        "model": args.version,
        "out_dim": 256,  
        "seg_token_idx":151665,
        "torch_dtype": torch_dtype,
        "attention":"flash_attention_2"
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VlmSamSegForCausalLM(config, **model_args).to(device)

    lora_target_modules = args.lora_target_modules.split(",")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.vlm = get_peft_model(model.vlm, lora_config)
    

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.version)
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    if num_added_tokens > 0:
        model.vlm.resize_token_embeddings(len(tokenizer))

    print(f"Loading checkpoint from {args.resume} ...")
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = remove_module_prefix(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing Keys:", missing_keys)
    print("Unexpected Keys:", unexpected_keys)


    if hasattr(model.vlm, "merge_and_unload"):
        print("Merging LoRA weights into the base model...")
        model.vlm = model.vlm.merge_and_unload()
    else:
        raise AttributeError("The model does not have merge_and_unload() method. "
                             "Make sure you are using a PEFT version that supports merging LoRA adapters.")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Merged model saved at {args.save_path}")

if __name__ == "__main__":
    main()
