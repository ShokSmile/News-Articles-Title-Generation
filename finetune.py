from utils.init_model import *
from utils.dataset import TitleDataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
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

## parameters
SPECIAL_TOKEN = "<|TG|>"

os.TOKENIZERS_PARALLELISM = False




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model finetuning for kaggle competition")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="challenge_files/data/train.csv",
        help="Path to unpreprocessed training data",
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
        default="challenge_files/data/validation.csv",
        help="Path to unpreprocessed validation data",
    )
    parser.add_argument(
        "--load_in_8bit", type=bool, default=True, help="If use 8bit quantization"
    )
    parser.add_argument(
        "--load_in_4bit", type=bool, default=False, help="If use 4bit quantization"
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
    parser.add_argument(
        "--hub_token", type=str, default="", help="HF hub token"
    )
    parser.add_argument("--batch_size_train", type=int, default=1, help="Size of batch for training")
    parser.add_argument("--batch_size_val", type=int, default=1, help="Size of batch for validation")
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
        "--model_dir", type=str, default='t5-small', help="Directory to save model"
    )
    parser.add_argument(
        "--name_of_wandb_project", type=str, default="INF582 2024 - News Articles Title Generation"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=712, help="Model input max length"
    )
    parser.add_argument("--bf16", type=bool, default=False, help="Use bf16 type in case of Ampere gpu architecture")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Model name")

    parser.add_argument("--predict_with_generate",
                        type=bool,
                        default=True,
                        help="During evaluation we generate outputs using model.generate(...) for the Seq2Seq models like T5 is already an argument for HF Trainer")


    parser = deepspeed.add_config_arguments(parser)
    sys_arg = parser.parse_args()
    sys_arg.deepspeed = True
    logging.getLogger().setLevel(logging.INFO)
    
    os.environ["WANDB_PROJECT"] = sys_arg.name_of_wandb_project
    
    # seed fixation
    np.random.seed(sys_arg.seed)
    torch.manual_seed(sys_arg.seed)
    torch.cuda.manual_seed(sys_arg.seed)
    
    
    #tokenizer initialization and 
    if sys_arg.local_rank == 0:
        logging.info("Tokenizer and data collator initialization...")
    
    tokenizer = AutoTokenizer.from_pretrained(sys_arg.model_checkpoint, 
                                              model_max_length=sys_arg.model_max_length,
                                              truncation=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [SPECIAL_TOKEN]}
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    #data initialization
    if sys_arg.local_rank == 0:
        logging.info("Preparing training and validation datasets ... ")
    

    train_data = TitleDataset(tokenizer=tokenizer,
                              data_path=sys_arg.train_data_path)
    
    val_data = TitleDataset(tokenizer=tokenizer, 
                              data_path=sys_arg.validation_data_path)
    
    #training args preparing
    if sys_arg.local_rank == 0:
        logging.info("Preparing training arguments ...")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=sys_arg.model_dir,
        run_name=sys_arg.model_name,
        evaluation_strategy=sys_arg.evaluation_strategy,
        eval_steps=sys_arg.eval_steps,
        logging_strategy="steps",
        logging_steps=sys_arg.eval_steps,
        save_strategy="steps",
        push_to_hub=sys_arg.push_to_hub,
        hub_model_id=sys_arg.model_name,
        hub_token=sys_arg.hub_token,
        save_steps=sys_arg.eval_steps,
        learning_rate=4e-5,
        per_device_train_batch_size=sys_arg.batch_size_train,
        per_device_eval_batch_size=sys_arg.batch_size_val,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=sys_arg.nb_epochs,
        predict_with_generate=sys_arg.predict_with_generate,
        group_by_length=True,
        generation_max_length=100,
        bf16=sys_arg.bf16,
        load_best_model_at_end=True,
        report_to="wandb",
        gradient_checkpointing=sys_arg.gradient_checkpointing,
        metric_for_best_model="eval_rougeL",
        seed=7,
        data_seed=7,
        # deepspeed="default_offload_opt_param.json", only for multigpus
    )
    
    if sys_arg.local_rank == 0:
        logging.info("Preparing model ...")
        
    if sys_arg.peft:
        if sys_arg.pretrained_peft:
            model = load_pretrained_peft_model(
                path_to_model_weights=sys_arg.model_checkpoint,
                load_in_8bit=sys_arg.load_in_8bit,
                load_in_4bit=sys_arg.load_in_4bit,
                path_to_peft_weights=sys_arg.path_to_peft_weights,
            )
        else:
            model = load_peft_model(
                path_to_weights=sys_arg.model_checkpoint,
                load_in_8bit=sys_arg.load_in_8bit,
                load_in_4bit=sys_arg.load_in_4bit,
            )

    else:
        model = load_ordinary_model(
            path_to_model_weights=sys_arg.model_checkpoint,
            load_in_8bit=sys_arg.load_in_8bit,
            load_in_4bit=sys_arg.load_in_4bit
        )
        
    if sys_arg.local_rank == 0:
        logging.info("Preparing trainer ...")
    
    metric = load_metric("rouge") 
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator, 
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(sys_arg.model_dir)