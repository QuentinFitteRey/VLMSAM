# Qwen-SAM: Reasoning Segmentation Model

## Introduction

This project presents **Qwen-SAM**, a reasoning segmentation model that leverages the power of the **Qwen 2.5 VL 7B** large multi-modal model and the **Segment Anything Model (SAM)**. It is a significant modification of the original [LISA (Language-Image Segment-Anything)](https://github.com/dvlab-research/LISA) framework, adapted to create a more robust system capable of understanding complex prompts and performing fine-grained segmentation based on reasoning.

The core idea is to combine the strong visual grounding and segmentation capabilities of SAM with the advanced language understanding and reasoning abilities of Qwen 2.5 VL. The model is trained to take an image and a textual prompt (e.g., "Can you segment the parked car?") and output a corresponding segmentation mask. LoRA is utilized for efficient fine-tuning of the Qwen model's text layers.

---
## Key Features

* **Reasoning-based Segmentation**: Designed to understand and act upon complex, descriptive, or reasoning-based textual prompts.
* **Qwen 2.5 VL Integration**: Utilizes the Qwen 2.5 VL 7B model for advanced text and image understanding.
* **SAM for Segmentation**: Employs a frozen SAM encoder for robust visual feature extraction and a trainable SAM decoder for generating segmentation masks.
* **Distributed Training**: Supports Distributed Data Parallel (DDP) for multi-GPU training.

---
## File Structure

A brief overview of the important files and directories:

```text
├── dataset/                  # Root directory for all datasets
├── model/
│   └── vlmsam.py             # Contains the VlmSamSegForCausalLM model definition
├── weights/                  # Directory for pretrained weights (e.g., SAM)
├── runs_*/                   # Output directory for training runs, logs, and checkpoints
├── utils/
│   ├── dataset.py            # Dataset handling and loading logic
│   └── utils.py              # Utility functions
├── .gitignore
├── requirements.txt          # Python dependencies
├── train_ddp.py              # Main script for training the model with DDP
├── test.py                   # Script for evaluating a trained model
├── inference.py              # Script for running inference with a trained model
├── transform_weight.py       # Merge trained model
```
---
## Installation

1.  **Prerequisites**:
    * Python 3.8+
    * PyTorch (refer to official website for version compatible with your CUDA toolkit)
    * CUDA Toolkit

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/QuentinFitteRey/VLMSAM
    cd VLMSAM
    ```

3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---
## Dataset

This model is trained on a mixture of datasets to support various segmentation tasks including semantic segmentation, referring segmentation, visual question answering, reasoning segmentation, and chain-of-thought style data.

**Supported Datasets** (as per `train_ddp.py` arguments):
* **Semantic Segmentation**: `ade20k`, `cocostuff`, `pascal_part`, `paco_lvis`, `mapillary`
* **Referring Segmentation**: `refclef`, `refcoco`, `refcoco+`, `refcocog`
* **VQA**: `llava_instruct_150k`
* **Reasoning Segmentation**: `ReasonSeg` (custom format, e.g., `ReasonSeg|train`, `ReasonSeg|val`, `ReasonSeg|test`)
* **Chain-of-Thought/Instruction**: `caption`, `cot`, `conversation`

**Data Preparation**:
1.  Download the required public datasets. You can adapt the download instructions from the original LISA project's for common datasets.
2.  Organize your datasets under the `dataset_dir` (default: `./dataset/`) specified during training/evaluation. The expected structure is generally:

```text
./dataset/
├── ade20k/
│   ├── images/
│   └── annotations_instance/
├── cocostuff/
│   ├── images/
│   └── annotations/
├── coco2017/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
├── mapillary/
│   ├── training/
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
├── refcoco/
│   └── annotations/
├── refcoco+/
│   └── annotations/
├── refcocog/
│   └── annotations/
├── refclef/
│   ├──saiapr_tc-12/
│   │  ├── images/
│   │  └── ...
│   └── annotations/
├── reason_seg/
│   ├── train/
│   ├── val/
│   └── test/
├── llava_dataset/
│   └── llava_v1_5_mix665k.json
├── textvqa/
│   ├── train_images/
├── ocr_vqa/
│   ├── images/
├── gqa/
│   ├── images/
├── seg++/
│   ├── caption.arrow
│   ├── conversation.arrow
│   ├── cot.arrow
│   ├── instance_seg.arrow
├── vg/
│   ├──VG_100K
│   ├──VG_100K_2
├── vlpart/
│   ├── paco/
│   │   └── annotations/
│   ├── pascal_part/
└── 
```

**Note on VQA Datasets:**
Visual instruction tuning relies on `llava_v1_5_mix665k.json` (located in `llava_dataset/`). The `textvqa/`, `ocr_vqa/`, and `gqa/` datasets are also mandatory and are used in conjunction with `llava_v1_5_mix665k.json` to enhance the model's visual understanding, reasoning, and Q&A capabilities, which complements its core segmentation functions.

While the larger `llava_v1_5_mix665k.json` is aimed at higher performance, a smaller dataset like `llava_instruct_150k.json` (another LLaVA variant) might be considered for a simpler setup or quicker training iterations. However, it's generally expected that using such a reduced dataset would lead to lower performance compared to the more extensive data mix.

---
## Pretrained Weights

1.  **Qwen 2.5 VL Model**:
    The model uses a pretrained Qwen VL model specified by the `--version` argument (e.g., `Qwen/Qwen2.5-VL-7B-Instruct`). This will be automatically downloaded from Hugging Face Hub by the `transformers` library during the first run if not cached.

2.  **SAM Vision Encoder**:
    The SAM vision encoder weights need to be downloaded separately. The training script expects them at the path specified by `--vision_pretrained` (default: `./weights/sam_vit_h_4b8939.pth`).
    You can typically find SAM weights (e.g., `sam_vit_h_4b8939.pth`) from official SAM releases. Ensure you place the downloaded file in the `./weights/` directory or update the path in your training command.

3.  **LoRA / Full Model Checkpoints**:
    * During training, checkpoints containing LoRA weights (if LoRA is enabled) and other trainable parts of the model are saved.
    * For evaluation or inference, you will use these saved checkpoints.

---
## Training

The model is trained using Distributed Data Parallel (DDP).

**Command**:
```bash
torchrun --nproc_per_node=<number_of_gpus> train_ddp.py [OPTIONS]
```
For example, to train on a single GPU:
```bash
torchrun --nproc_per_node=1 train_qwen_ddp.py \
    --version "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_dir "./dataset" \
    --log_base_dir "./runs_qwen/experiment_001" \
    --exp_name "qwen_sam_reasonseg" \
    --epochs 10 \
    --batch_size 2 \
    --grad_accumulation_steps 8 \
    --lr 3e-5 \
    --lora_r 16 \
    --precision "bf16" \
    --vision_pretrained "./weights/sam_vit_h_4b8939.pth" \
    --dataset "reason_seg||refer_seg" \
    --sample_rates "1,1" \
    --reason_seg_data "ReasonSeg|train" \
    --refer_seg_data "refcoco||refcoco+||refcocog" \
    --val_dataset "ReasonSeg|val"
    # Add other arguments as needed
```
---
### Checkpointing & Resumption

Model **checkpoints are saved periodically** during training to the directory specified by `args.log_base_dir`. To resume a previous training run, use the `--resume` argument followed by the path to your checkpoint file:

```bash
--resume <path_to_checkpoint.pth>
```

---
## Evaluation (Testing)

To evaluate a trained model on a validation or test set using a single GPU: 🧐

**Command**:
You can directly use `python` to run the evaluation script if it's designed to handle single GPU execution (which the `test.py` script you provided does by checking environment variables or defaulting `local_rank`).

```bash
python evaluate_qwen_ddp.py \
    --resume <path_to_your_checkpoint.pth> 

```

## Inference 💡

Running inference with your trained Qwen-SAM model typically involves two main steps:

1.  **Prepare Model for Inference (Merge LoRA Weights)**:
    If you trained your model using LoRA (Low-Rank Adaptation), the learned adaptations are stored separately from the base model weights in your checkpoints. For easier and potentially more efficient inference, it's recommended to merge these LoRA weights directly into the base model. The provided script (`transform_weight.py`) handles this.

2.  **Run Inference Script**:
    Once the weights are merged, you can use a standard inference script (`inference.py`) that loads this complete model for performing segmentation tasks.

---
### Step 1: Merge LoRA Weights

Use the `transform_weight.py` script to combine the LoRA adapters from your training checkpoint with the base model weights.

**Command**:
```bash
python transform_weight.py \
    --version "Qwen/Qwen2.5-VL-7B-Instruct" \
    --resume "/path/to/your/training_checkpoint.pth" \
    --save_path "/path/to/save/merged_model_statedict.pth" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj,v_proj" \
    --precision "bf16"
```
Ensure LoRA parameters (--lora_r, --lora_alpha, etc.) match those used for training the checkpoint.

### Step 2: Run Inference using `inference.py`

Once you have a model checkpoint (ideally the merged one from Step 1, or your final training checkpoint if LoRA was not used/already merged), you can use `inference.py` to perform segmentation on new images with text prompts.

This script will load the specified model checkpoint, allow you to input an image path and a prompt, and then save the generated segmentation mask and an overlaid image.

**Command**:
```bash
python inference.py \
    --version "Qwen/Qwen2.5-VL-7B-Instruct" \
    --resume "/path/to/your/merged_model_for_inference.pth" \
    --vis_save_path "./vis_output" \
    --precision "bf16" \
    --image_size 1024 \
    --model_max_length 2048
```

**Interactive Usage**:
After running the command, the script will prompt you in the console:
1.  `Please input your prompt (or press Enter to exit):`
2.  `Please input the image path or URL:`

Enter the requested information to get segmentation results. Outputs will be saved in the directory specified by `--vis_save_path`.

**Output**:
The script will print the decoded segmentation text and save:
* The predicted segmentation mask as an image (e.g., `imagename_mask_0.jpg`).
* The original image with the segmentation mask overlaid (e.g., `imagename_masked_img_0.jpg`).
Paths to these saved files will be printed in the console.


---
## Acknowledgements

* This work is  based on the **LISA (Language-Image Segment-Anything)** project. Thanks to the original authors for their work.
* We acknowledge the **Qwen Team (Alibaba Cloud)** for developing and releasing the Qwen Large Vision Language Models (VLMs) that are a core component of this project.
* Our work utilizes the **Segment Anything Model (SAM)**, and we thank **Meta AI** for their groundbreaking work in image segmentation and for making SAM publicly available.