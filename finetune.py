from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
    PeftConfig,
)
import numpy as np
import pandas as pd
from datasets import load_metric
import torch
import re
import os
import argparse
import deepspeed
import logging

###########################################
# Parameters
MODEL_CHECKPOINT = "t5-large"
REAL_FILE_PATH = "Data/real_prompt_data.json"
REAL_SYN_FILE_PATH = "Data/real-syn_prompt_data.json"
PROBLEM_RETAINED_PATH = "Data/problem_retained_clusters.json"
# right now it doesn't matter cause we fixed 25 samples (hardcode)
VAL_SIZE = 0.05
BATCH_SIZE = 16
EVAL_STEPS = 100
EPOCHS = 300
INCLUDE_SYNTHETIC = False
OUTPUT_MODE = "problem"
MODEL_NAME = "113ex-dpn-desc" + OUTPUT_MODE + "-" + MODEL_CHECKPOINT.replace("ShokSmile/", "")
MODEL_DIR = f"Models/{MODEL_NAME}"
MODEL_MAX_LENGTH = 1024
INCLUDE_DESCRIPTION = True
TOKEN_PROBLEM = "<|pr|>"
TOKEN_ROOT_CAUSE = "<|rc|>"
TOKEN_SENTENCE = "<|sentence|>"
SYN_TOKEN = "<syn>"
PEFT = False
LORA = False
LoRA_PRETRAINED = False
LoRA_PATH = "<path to be changed>"
seed = 7
os.environ["WANDB_PROJECT"] = "INF582 2024 - News Articles Title Generation"
############################################

# numpy seed fixation
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    use_fast=False,
    model_max_length=MODEL_MAX_LENGTH,
    use_auth_token="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",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model finetuning for kaggle competition")
    parser.add_argument(
        "--data_path",
        type=str,
        default="challenge_files/data/train.csv",
        help="Path to unpreprocessed data",
    )
    parser.add_argument(
        "--load_in_8bit", type=bool, default=True, help="If use 8bit quantization"
    )
    parser.add_argument(
        "--model_checkpoint", type=str, default="t5-small", help="Path to model checkpoint"
    )
    parser.add_argument("--peft", type=bool, default=True, help="Use PEFT")
    parser.add_argument(
        "--pretrained_peft",
        type=bool,
        default=False,
        help="Fine tuning already pretrained PEFT",
    )
    parser.add_argument(
        "--path_to_peft_weights",
        type=str,
        default="",
        help="Path to PEFT adapters weights",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Every eval_steps we have evaluation based on val_set",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="In case of DDP")
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy (steps|epochs)",
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        default=True,
        help="Use HF Hub to save model weights",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Size of batch")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Gradient checkpoints usage",
    )
    parser.add_argument(
        "--nb_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed")
    parser.add_argument(
        "--model_dir", type=str, default='../Model/llama-2-7b-dpn-lisi-ponticelli-peft-v1', help="Directory to save model"
    )
    parser.add_argument(
        "--name_of_wandb_project", type=str, default="INF582 2024 - News Articles Title Generation"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=1024, help="Model input max length"
    )
    parser.add_argument("--bf16", type=bool, default=False, help="Use bf16 type in case of Ampere gpu architecture")
    parser.add_argument("--model_name", type=str, default="", help="Model name")

    parser.add_argument("--predict_with_generate",
                        type=bool,
                        default=True,
                        help="During evaluation we generate outputs using model.generate(...) for the Seq2Seq models like T5 is already an argument for HF Trainer")


    parser = deepspeed.add_config_arguments(parser)
    sys_arg = parser.parse_args()
    sys_arg.deepspeed = True
    logging.getLogger().setLevel(logging.INFO)