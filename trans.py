#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from dschat.utils.model.model_utils import create_hf_model
from dschat.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from dschat.utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
)
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)

from transformers import AutoTokenizer

import pdb  
from reward_model_test import RewardModelTransOutput as RewardModel # 
from transformers import (
    AutoConfig,
    AutoModel,
)

import pickle
from pathlib import Path

def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        dropout=None,
                        zero_stage=0,
                        compute_fp32_loss=False,
                        enlarge_factor=1):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, dropout)
    end = time.time()
    print_rank_0(f">Creating model from_config took {end - start} seconds",
                 None)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss,
        enlarge_factor=enlarge_factor)
    return critic_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `2,4,4`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    parser.add_argument(
        "--l1_lambda", type=float, default=0.0, help="l1 reg to use."
    )

    parser.add_argument(
        "--enlarge_factor", type=float, default=1.0, help="enlarge factor for the linear mapping, contractive if < 1.0"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.",
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use.",
    )

    # Evaluation
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="If > 0, perform evaluation at this interval",
    )
    parser.add_argument(
        "--eval_iters", type=int, default=100, help="Maximum evaluation iterations"
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32.",
    )

    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step2_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action="store_true",
        help="Add <|endoftext|> as additional special token to tokenizer",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    print_rank_0(f"[Debug]: accelerator={get_accelerator()} device={device}")
    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step2_model",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )
    print_rank_0(f"[Debug]: ds_config={ds_config}")

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = (
        args.end_of_conversation_token if args.add_eot_token else None
    )
    args_dict = vars(args)
    with open(args.output_dir + "/args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, fast_tokenizer=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = "right"
    
    # dschat.utils.model.reward_model.RewardModel
    rm_model = create_critic_model(
        args.model_name_or_path,
        tokenizer,
        ds_config,
        args.num_padding_at_beginning,
        dropout=args.dropout,
        zero_stage=args.zero_stage,
        compute_fp32_loss=args.compute_fp32_loss,
        enlarge_factor=args.enlarge_factor
    )

    # Model bigscience/bloom-560m has large variance at ln_f.weight parameter
    # This makes bf16 finetuning hard.
    # In general, since we are replacing the model head, it makes sense to reset
    # the LN that precedes it.
    force_optimize_params = []
    if "bigscience/bloom-" in args.model_name_or_path:
        zero_init_enabled = args.zero_stage == 3
        params = [
            rm_model.rwtranrsformer.ln_f.weight,
            rm_model.rwtranrsformer.ln_f.bias,
        ]
        with deepspeed.zero.GatheredParameters(
            params, modifier_rank=0, enabled=zero_init_enabled
        ):
            if deepspeed.comm.get_rank() == 0 or not zero_init_enabled:
                torch.nn.init.ones_(rm_model.rwtransformer.ln_f.weight)
                torch.nn.init.zeros_(rm_model.rwtransformer.ln_f.bias)
        force_optimize_params.extend(
            ["rwtransformer.ln_f.weight", "rwtransformer.ln_f.bias"]
        )

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(
            rm_model, args.lora_module_name, args.lora_dim
        )
        if args.only_optimize_lora:
            force_optimize_params.append("v_head.weight")
            rm_model = only_optimize_lora_parameters(rm_model, force_optimize_params)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
    )

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )

    def evaluation_reward(model, dataloader, eval_iters):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        chosen_scores = 0.0
        rejected_scores = 0.0
        chosen_features_eval_list = []
        rejected_features_eval_list = []
        for _step, _batch in enumerate(dataloader):
            _batch = to_device(_batch, device)
            with torch.no_grad():
                _outputs = model(**_batch)

            chosen_features_eval = _outputs["chosen_features"]
            rejected_features_eval = _outputs["rejected_features"]
            chosen_features_eval_list.append(chosen_features_eval)
            rejected_features_eval_list.append(rejected_features_eval)
            
        return chosen_features_eval_list, rejected_features_eval_list


    for param in rm_model.parameters():
        param.requires_grad = False
    last_layer_index = rm_model.config.num_hidden_layers - 1
    
    parameters_to_l1regularize = ["v_head.weight"]

    # Iterate over the model parameters
    for name, param in rm_model.named_parameters():
        if name in parameters_to_l1regularize:
            param.requires_grad = True
    
    def print_l1regularized_named_parameters(model):
        l1regularized_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and name in parameters_to_l1regularize
        ]
        print_rank_0(f"Number of l1 regularized parameters: {len(l1regularized_params)}")
        for name, param in l1regularized_params:
            print_rank_0(f"Parameter name: {name}, Shape: {param.shape}")
        return l1regularized_params
    
    l1regularized_params = print_l1regularized_named_parameters(rm_model)

    parameters_to_l12regularize = ["linear_mapping.weight"]
    for name, param in rm_model.named_parameters():
        if name in parameters_to_l12regularize:
            param.requires_grad = True

    def print_l12regularized_named_parameters(model):
        l12regularized_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and name in parameters_to_l12regularize
        ]
        print_rank_0(f"Number of l12 regularized parameters: {len(l12regularized_params)}")
        for name, param in l12regularized_params:
            print_rank_0(f"Parameter name: {name}, Shape: {param.shape}")
        return l12regularized_params

    l12regularized_params = print_l12regularized_named_parameters(rm_model)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate
    )

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    # Filter the parameter groups
    filtered_optimizer_grouped_parameters = [
        {"params": filter(lambda p: p.requires_grad, group["params"]), **{k: v for k, v in group.items() if k != "params"}}
        for group in optimizer_grouped_parameters
    ]
    optimizer = AdamOptimizer(
        filtered_optimizer_grouped_parameters, 
        lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank,
    )

    chosen_features_eval_list, rejected_features_eval_list = evaluation_reward(
        rm_model, eval_dataloader, args.eval_iters
    )

    total_micro_steps = 0
    for epoch in range(1):
        chosen_features_list = []
        rejected_features_list = []
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        rm_model.eval()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            chosen_features = outputs["chosen_features"]
            rejected_features = outputs["rejected_features"]
            chosen_features_list.append(chosen_features)
            rejected_features_list.append(rejected_features)
        
        output_dir = Path(args.output_dir)
        data_part = ""
        for i in range(len(args.data_path)):
            data_part += str(args.data_path[i]).split("/")[1]
        model_part = str(args.model_name_or_path).split("/")[1]
        file_path = output_dir / (data_part + model_part)
        with open(file_path, "wb") as f:
            pickle.dump((chosen_features_list, rejected_features_list, chosen_features_eval_list, rejected_features_eval_list), f)



if __name__ == "__main__":
    main()
