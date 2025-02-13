# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import defaultdict
import os
import copy
import sys
from torch_geometric.utils import k_hop_subgraph
from torch_cluster import random_walk
import spacy
import torch_geometric

sys.path.append("./")
sys.path.append("./utils")

from utils.metric import analyze_composition
from dataclasses import dataclass, field
import json
import logging
import pathlib
import pickle
import re
from typing import Dict, Optional, Sequence
import pandas as pd

from sklearn.metrics import cohen_kappa_score
import torch
from termcolor import colored
import transformers
import torch.nn.functional as F
from utils.constants import IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
from torch.utils.data import Dataset
from llaga_trainer import LLaGATrainer

from model import *
from torch_geometric.utils import subgraph
import random
from tqdm import trange
from utils import conversation as conversation_lib
from utils.utils import tokenizer_graph_token
import scipy.sparse as sp
import numpy as np
from ChildText.prompt import GOLD_TEXT, LABEL_TEMPLATE_SCORE, LABEL_TEMPLATE_TRAIT, LABEL_TEMPLATE_TRAIT_COMMENT,PROMPT_TEMPLATE, PROMPT_TEMPLATE_GRAPH_MATCH, PROMPT_TEMPLATE_TRAIT_COMMENT, PROMPT_TEMPLATE_TRAIT_ONLY_GRAPH,PROMPT_TEMPLATE_V1,PROMPT_TEMPLATE_only_graph

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_graph_encoder_weights: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=True)
    tune_graph_encoder: bool = field(default=True)
    graph_encoder_type: Optional[str] = field(default=None,metadata={"help":"GraphAttModel or GAT"})
    graph_encoder_num_hidden: Optional[int] = field(default=2048)
    graph_encoder_num_heads: Optional[int] = field(default=4)
    graph_encoder_num_layers: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    pretrained_embedding_type: Optional[str] = field(default='sbert')
    use_hop: Optional[int] = field(default=2)
    sample_neighbor_size: Optional[int] = field(default=-1)
    use_task:Optional[str] = field(default="nc")
    use_dataset:Optional[str] = field(default="ChildText")
    template: Optional[str] = field(default="ND")
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    is_only_graph: bool = field(default=False)
    is_trait_comment: bool = field(default=False)
    is_trait: bool = field(default=False)
    is_precompute_micro_metric: bool = field(default=True)
    is_edge_str: bool = field(default=True)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)



def verify_model_dtype(model):
    """
    功能: 查看模型中各种类型的参数的情况
    """
    rank0_print(f"--> model structure: \n{model}")
    ignore_layers = [f"layers.{i}" for i in range(2,21)]  # 减少打印的层数
    rank0_print(f"ignore print layers: \n{ignore_layers}")
    for n,v in model.named_parameters():
        # 少打印一些层
        if not any([i in n for i in ignore_layers]):
            if v.requires_grad:
                rank0_print(f"trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")
            else:
                rank0_print(f"not trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")
 
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype   # 获取参数的数据类型
        # 统计参数数量和参数名称
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        # 如果参数参与训练(requires_grad=True),则统计可训练参数的数量和名称
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    rank0_print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        rank0_print("all params info: {}  num: {}  {:.3f}%".format(k, v, 100.0 * v / total))  # 打印每种数据类型的参数量和占比
    rank0_print()
    
    # 统计可训练参数中，各种类型参数分布
    rank0_print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        rank0_print("trainable params info: {}  num: {}  {:.3f}%".format(k, v, 100.0 * v / total_trainable))
    rank0_print()
    for k, v in dtype2trainable_param_name.items():
        rank0_print("all params info: {}  trainable layers: {}".format(k, v))   # 打印每种数据类型的可训练参数名称
    rank0_print()
    # 查看参与训练的参数情况
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print("Total model params: %.2fM" % (total / 1e6))
    rank0_print(
        f'trainable params: {trainable} || all params: {total} || trainable%: {round(trainable / total, 4)}')
 
 


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_graph_encoder_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        mm_projector_keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_graph_start_end", False):
            mm_projector_keys_to_match.extend(['embed_tokens', 'embed_in'])

        mm_projector_weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), mm_projector_keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(mm_projector_weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(mm_projector_weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        if getattr(trainer.args, "tune_graph_encoder", False):
            graph_encoder_keys_to_match = ['graph_encoder']
            graph_encoder_weight_to_save = get_graph_encoder_state_maybe_zero_3(trainer.model.named_parameters(), graph_encoder_keys_to_match)
            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    graph_encoder_folder = os.path.join(parent_folder, "graph_encoder")
                    os.makedirs(graph_encoder_folder, exist_ok=True)
                    torch.save(graph_encoder_weight_to_save, os.path.join(graph_encoder_folder, f'{current_folder}.bin'))
                else:
                    torch.save(graph_encoder_weight_to_save, os.path.join(output_dir, f'graph_encoder.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3
    roles = {"human": "user", "gpt": "assistant"}
    sep_start = "<|start_header_id|>"
    sep_end = "<|end_header_id|>"
    eot_token = "<|eot_id|>"

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = [] # 合并输入信息："A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? ASSISTANT: Genetic_Algorithms</s>"
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,# Genetic_Algorithms
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_graph_token(rou, tokenizer)) + len(tokenizer_graph_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_graph_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("v1"):# 运行这个
        return preprocess_v1(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_graph_token(prompt, tokenizer)) for prompt in prompts]

    if has_graph:
        input_ids = [tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_graph:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def build_heterogeneous_graph_string(edge_index, edge_type):
    """
    构建异构图的邻接表并输出为字符串形式，将边类型映射为类型名称。
    
    参数:
    - edge_index: list 或 array，形状为 (2, N)，表示 N 条边的起始和终点节点。
    - edge_type: list 或 array，长度为 N，对应每条边的类型。
    
    返回:
    - result: str，表示图的邻接表，其中每个节点对应一个邻接列表，
              邻接列表中的每个元素是 (邻居节点, 边类型)。
    """
    # 边类型的映射
    edge_type_mapping = {
        0: '使能-因果',          # enabling-causal
        1: '动机-因果',          # psychological-causal
        2: '物理-因果',          # motivational-causal
        3: '心理-因果',          # physical-causal
        4: '并列'               # parallel
    }

    # 初始化邻接表
    graph = {}

    # 遍历所有边，构建邻接表
    for i in range(len(edge_index[0])):
        src = edge_index[0][i]  # 起点节点
        dest = edge_index[1][i] # 终点节点
        e_type = edge_type[i]   # 边的类型
        if e_type == -1:
            #FIXME 如果没有匹配的边类型项，etype=-1   
            print("没有匹配的边类型项，etype=-1")
            continue
        # 获取该边的类型名称
        edge_label = edge_type_mapping[e_type.item()]
        
        # 如果源节点还不在图中，初始化它的邻接表
        if src not in graph:
            graph[src] = []

        # 添加边到源节点的邻接表
        graph[src].append((dest, edge_label))
    
    # 构建输出字符串
    result = ""
    for node, neighbors in sorted(graph.items()):
        neighbor_str = ", ".join([f"({neighbor[0]}, '{neighbor[1]}')" for neighbor in neighbors])
        result += f"Node {node}: [{neighbor_str}],"
    
    return result.strip()  # 去掉最后一行的换行符

class LazySupervisedGraphDataset_ChildText(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 mode=None):
        super(LazySupervisedGraphDataset_ChildText, self).__init__()
        
        self.datas={}
        dataset = 'ChildText'
        if mode == "train":
            data_path = data_args.data_path
        else:
            data_path = data_args.eval_data_path
        rank0_print(f"{mode} dataset :Load data from :",data_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.datas[dataset]=data

        self.use_task = data_args.use_task.split('-')
        # 加载中文预训练模型（zh_core_web_sm）
        if data_args.is_precompute_micro_metric:
            nlp_model = spacy.load("zh_core_web_sm")
        for task in self.use_task:
            if task == "graph_score":
                for id,g in enumerate(data):
                    if data_args.is_only_graph:         # 只输入叙事图，只做评分
                        prompt = PROMPT_TEMPLATE_TRAIT_ONLY_GRAPH
                    elif data_args.is_trait_comment or data_args.is_trait :    # 叙事图+文本，多维度评分加评论
                        prompt = PROMPT_TEMPLATE_TRAIT_COMMENT
                                              
                    else:                               # 叙事图+文本，只做评分       
                        prompt = PROMPT_TEMPLATE
                    prompt = prompt.replace("<essay_text>",g.essay_text)
                    if data_args.is_precompute_micro_metric:
                        micro_metric = analyze_composition(text=g.essay_text,nlp=nlp_model)
                        prompt = prompt.replace("<micro_metrics_str>",micro_metric)
                    else:
                        prompt = prompt.replace("3. 微观结构维度评分时，请你使用以下量化数据作为参考：<micro_metrics_str>\n","")

                    if data_args.is_trait_comment:
                        label_template = LABEL_TEMPLATE_TRAIT_COMMENT
                    if data_args.is_trait:
                        label_template = LABEL_TEMPLATE_TRAIT
                    else:
                        label_template = LABEL_TEMPLATE_SCORE
                    prompt = prompt.replace("<label_template>",label_template)
                    
                    if data_args.is_edge_str:
                        edge_code_str = build_heterogeneous_graph_string(g.edge_index,g.edge_type)
                        prompt = prompt.replace("<edge_code_str>",edge_code_str)
                    else:
                        prompt = prompt.replace("- 边结构：<edge_code_str>\n",'')
                        
                    g.id = id
                    g.dataset = dataset

                    labels = make_label(data_args=data_args,g=g)
                    g.conversations = [{"from":"human","value":prompt},
                                       {"from":"gpt","value": labels}
                                       ]
            #print(prompt)
            rank0_print(f"************* **{mode}** Dataset {dataset} ,Task {task}, size {len(data)}****************")
            #list_data_dict.extend(task_list_data_dict) # len=1624

        # 处理标签

        if mode  == 'train':
            rank0_print("node embed size:",g.x.size(1))
            rank0_print(colored(data[10].conversations,'green'))
            rank0_print(f"Formatting inputs...Skip in lazy mode, size {len(data)}")
        self.tokenizer = tokenizer
        #self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data = data
        # if mode == 'val':
        #     self.data = self.data[10]
        random.shuffle(self.data)



    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        sources = copy.deepcopy(self.data[i].conversations) # from human;from gpt
        sources = [sources]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_graph=True
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],# 这里的input还没有graph token
                             labels=data_dict["labels"][0])
        data_dict['graph'] = torch.LongTensor(range(self.data[i].x.size(0))).unsqueeze(0)#  【1,34】
        data_dict['graph_emb'] = self.data[i].x.unsqueeze(0)# [1,node-num,512]
        data_dict['edge_index'] = self.data[i].edge_index# 
        data_dict['edge_attr'] = self.data[i].edge_attr# 
        data_dict['edge_type'] = self.data[i].edge_type
        # if self.data_args.pretrained_embedding_type == "GraphAttModel":
        #     data_dict['g'] = self.data[i]
        # else:
        #     data_dict['g'] = None
        #label是token
        return data_dict#input_ids,labels,graph,graph_emb,


    def __len__(self):
        #return len(self.list_data_dict)
        return len(self.data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample['graphs']) if 'graphs' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'graph' in sample else -cur_len
            length_list.append(cur_len)
        return length_list


class SelfSupervisedGraphDataset_GraphMatch(Dataset):
    """Dataset for self supervised pretraining-graph_matching."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 mode=None):
        super(SelfSupervisedGraphDataset_GraphMatch, self).__init__()
        
        self.datas={}
        dataset = 'graph_match'
        if mode == "train":
            data_path = data_args.data_path
        else:
            data_path = data_args.eval_data_path
        rank0_print(f"{mode} dataset :Load data from :",data_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if mode == "val":
            data = data[:5] 
        self.datas[dataset]=data
        self.use_task = data_args.use_task
        match_data = []
        rank0_print(f"task={self.use_task}")
        if data_args.use_task == "graph_match":
            for id,g in enumerate(data):                      
                for _ in range(g.x.size(0) % 17):  #每个叙事图采样多个子图
                    # 调用函数生成子图
                    sampled_subgraph, original_text_sequence, shuffled_text_sequence = sample_random_subgraph(g)
                    if sampled_subgraph == None:
                        break
                    labels = text_sequence_to_label(original_text_sequence)
                    shuffled_text_sequence = text_sequence_to_label(shuffled_text_sequence)
                    prompt = PROMPT_TEMPLATE_GRAPH_MATCH
                    prompt = prompt.replace('<shuffled_text_sequence>',shuffled_text_sequence)
                    sampled_subgraph.id = id
                    sampled_subgraph.dataset = dataset
                    sampled_subgraph.conversations = [{"from":"human","value":prompt},
                                                    {"from":"gpt","value": labels}
                                                    ]
                    match_data.append(sampled_subgraph)
                    #rank0_print(sampled_subgraph)
            #print(prompt)
        rank0_print(f"************* **{mode}** Dataset {dataset} ,Task {self.use_task}, size {len(match_data)}****************")

        if mode  == 'train':
            # rank0_print("node embed size:",match_data[11].x.size(1))
            rank0_print(colored(match_data[199].conversations,'green'))
            # rank0_print(f"Formatting inputs...Skip in lazy mode, size {len(data)}")
        self.tokenizer = tokenizer
        #self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data = match_data
        random.shuffle(self.data)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        sources = copy.deepcopy(self.data[i].conversations) # from human;from gpt
        sources = [sources]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_graph=True
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],# 这里的input还没有graph token
                             labels=data_dict["labels"][0])
        data_dict['graph'] = torch.LongTensor(range(self.data[i].x.size(0))).unsqueeze(0)#  【1,34】
        data_dict['graph_emb'] = self.data[i].x.unsqueeze(0)# [1,node-num,512]
        data_dict['edge_index'] = self.data[i].edge_index# 
        data_dict['edge_attr'] = self.data[i].edge_attr# 
        data_dict['edge_type'] = self.data[i].edge_type
        # if self.data_args.pretrained_embedding_type == "GraphAttModel":
        #     data_dict['g'] = self.data[i]
        # else:
        #     data_dict['g'] = None
        #label是token
        return data_dict#input_ids,labels,graph,graph_emb,


    def __len__(self):
        #return len(self.list_data_dict)
        return len(self.data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample['graphs']) if 'graphs' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'graph' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

def text_sequence_to_label(text_list):
    formatted_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(text_list)])
    return formatted_str

def sample_random_subgraph(raw_data, initial_ratio=0.25, max_ratio=1.0):
    """
    从图中随机采样指定数量的节点，生成包含这些节点的子图，并更新原始图数据中的相关属性。
    如果采样的子图没有边，将逐步增加采样比例，直到形成有边的子图。
    
    参数：
    - data (torch_geometric.data.Data): 原始图数据，包括节点特征和边索引。
    - initial_ratio (float): 初始采样比例（如0.25表示采样总节点数的25%）。
    - max_ratio (float): 最大采样比例，防止无限增加采样数量。

    返回：
    - data (torch_geometric.data.Data): 更新后的图数据，包含采样的子图信息。
    - original_text_sequence (list): 原始节点文本序列。
    - shuffled_text_sequence (list): 打乱的节点文本序列。
    """
    data = copy.deepcopy(raw_data)
    total_nodes = data.num_nodes
    ratio = initial_ratio
    
    while ratio <= max_ratio:
        # 按当前比例采样节点，并按递增排序
        if total_nodes <=10:
            # 提取按递增排序的节点的文本序列
            original_text_sequence = [data.node_text[i] for i in range(total_nodes)]
            
            # 生成打乱的文本序列
            shuffled_text_sequence = original_text_sequence[:]
            random.shuffle(shuffled_text_sequence)
            # 返回更新后的 data 对象和文本序列
            return data, original_text_sequence, shuffled_text_sequence
        num_sampled_nodes = max(1, int(total_nodes * ratio))  # 至少采样一个节点
        sampled_nodes = torch.randperm(total_nodes-1)[:num_sampled_nodes]
        sampled_nodes = torch.sort(sampled_nodes).values  # 对采样的节点进行排序
        
        # 使用采样的节点生成子图
        sampled_edge_index, sampled_edge_attr, edge_mask = subgraph(
            sampled_nodes.cpu(),
            data.edge_index.cpu(),
            edge_attr=data.edge_attr.cpu(),
            relabel_nodes=True,
            return_edge_mask=True,
            num_nodes=total_nodes
        )
        
        # 检查子图是否包含边
        if sampled_edge_index.size(1) > 0:  # 如果子图有边
            # 更新 data 对象的节点特征和边信息
            data.x = data.x[sampled_nodes] if data.x is not None else None
            data.edge_index = sampled_edge_index
            data.edge_attr = sampled_edge_attr
            data.edge_type = data.edge_type[edge_mask.cpu()] if hasattr(data, 'edge_type') else None
            
            # 提取按递增排序的节点的文本序列
            original_text_sequence = [data.node_text[i] for i in sampled_nodes.cpu().tolist()]
            
            # 生成打乱的文本序列
            shuffled_text_sequence = original_text_sequence[:]
            random.shuffle(shuffled_text_sequence)
            
            # 返回更新后的 data 对象和文本序列
            return data, original_text_sequence, shuffled_text_sequence
        
        # 增加采样比例
        ratio += 0.1  # 每次增加10%的节点数
    
    # 如果最终没有找到包含边的子图，返回None
    return None, None, None

@dataclass
class DataCollatorForSupervisedDataset_ChildText(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # instances是list，batchsize个get—item的返回data_dict
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)# [batch-szie,128seq-len]
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph_emb' in instances[0]:# 
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
            edge_index_list = [instance['edge_index'] for instance in instances]
            edge_attr_list = [instance['edge_attr'] for instance in instances]
            edge_type_list = [instance['edge_type'] for instance in instances]
            # 找到批次中graph的最大node-num
            max_node_num = max([g.size(1) for g in graph])
            max_edges = max([adj.shape[1] for adj in edge_index_list])
            # 为graph添加填充，使得每个graph在node-num维度上对齐
            graph_padded = [torch.nn.functional.pad(g, (0, max_node_num - g.size(1)),
                                                    value=DEFAULT_GRAPH_PAD_ID).cpu()
                            for g in graph]

            # 为graph_emb添加填充，填充部分与graph的DEFAULT_GRAPH_PAD_ID对应，填充为emb=0
            graph_emb_padded = [torch.nn.functional.pad(ge, (0, 0, 0, max_node_num - ge.size(1)), value=0).cpu()
                                for ge in graph_emb]
            # TODO使用-1填充将邻接矩阵对齐
            padded_adj_list = [torch.nn.functional.pad(adj, (0, max_edges - adj.shape[1]), value=-1).cpu() for adj in edge_index_list]
            # 0填充边权重，使其边数一致
            padded_edge_attr_list = [torch.nn.functional.pad(edge_attr, (0, 0, 0, max_edges - edge_attr.shape[0]), value=0).cpu()
                                    for edge_attr in edge_attr_list]
            # 填充边类别，使其边数一致，默认填充值为 -1
            padded_edge_type_list = [torch.nn.functional.pad(edge_type, (0, max_edges - edge_type.shape[0]), value=-1).cpu()
                                        for edge_type in edge_type_list]
            batch['graph'] = torch.cat(graph_padded, dim=0).cpu()
            batch['graph_emb'] = torch.cat(graph_emb_padded, dim=0).cpu()
            batch['edge_index'] = torch.stack(padded_adj_list).cpu() # [num_graphs, 2, max_edges]
            batch['edge_attr'] = torch.stack(padded_edge_attr_list).cpu()# [num_graphs, max_edges, hidden]
            batch['edge_type'] = torch.stack(padded_edge_type_list).cpu()  # [num_graphs, max_edges]
            # batch_g = []
            # for instance in instances:
            #     instance['g'].x = instance['g'].x
            #     instance['g'].edge_attr = instance['g'].edge_attr
            #     batch_g.append(instance['g'])
            # batch['g'] = torch_geometric.data.Batch.from_data_list(batch_g).cpu()

        return batch
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #TODO instances是list，batchsize个get—item的返回data_dict
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)# [batch-szie,128seq-len]
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
            batch['graph'] = torch.cat(graph, dim=0)
            batch['graph_emb'] = torch.cat(graph_emb, dim=0)

        return batch

def rank0_print_input_ids(input_ids):
    input_clone = input_ids.clone()
    input_clone[input_clone == -200] = tokenizer.pad_token_id
    input_tmp = tokenizer.batch_decode(input_clone, skip_special_tokens=True)[0]
    rank0_print(input_tmp)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_cls = SelfSupervisedGraphDataset_GraphMatch if data_args.use_task == 'graph_match' else LazySupervisedGraphDataset_ChildText
    train_dataset = data_cls(tokenizer=tokenizer,data_args=data_args,mode='train')
    if data_args.eval_data_path is not None:
        eval_dataset = data_cls(tokenizer=tokenizer,data_args=data_args,mode='val')
    else:
        eval_dataset = None
    # train_dataset['test'].save_to_disk("./cora_tmp_test")
    data_collator = DataCollatorForSupervisedDataset_ChildText(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,# FIXME 没有传入测试数据集
                data_collator=data_collator)


def make_label(data_args,g):
    """
    构建标签string"""
    if data_args.is_trait_comment:
        labels = LABEL_TEMPLATE_TRAIT_COMMENT.replace("<comment>",g.comment).replace("<macro_score>",str(g.macro_score.item())).replace("<micro_score>",str(g.micro_score.item())).replace("<psych_score>",str(g.psych_score.item())).replace("<total_score>",str(g.total_score.item()))
    elif data_args.is_trait:
        labels = LABEL_TEMPLATE_TRAIT.replace("<macro_score>",str(g.macro_score.item())).replace("<micro_score>",str(g.micro_score.item())).replace("<psych_score>",str(g.psych_score.item())).replace("<total_score>",str(g.total_score.item()))    
    else:
        labels = "预测得分："+str(g.y.item())
    return labels


def extract_prediction_score(s):
    # 使用正则表达式匹配 "Prediction score is <数字>" 并提取数字
    match = re.findall(r'预测得分：(\d+)', s)

    if match:
        # 将匹配到的数字部分转换为整数
        return int(match[-1])
    else:
        # 如果没有匹配到则返回 None
        return None


# 定义 logits 的预处理函数
def preprocess_logits_for_metrics(logits, labels):
    """
    处理 LLaMA 的生成模型 logits，用于评估阶段。
    """
    # logits 的形状为 (batch_size, sequence_length, vocab_size)
    # 通常我们使用 argmax 转换 logits 为生成的 token 序列
    #rank0_print(logits.argmax(dim=-1))
    return logits.argmax(dim=-1)  # 返回 shape: (batch_size, sequence_length)


def compute_qwk(eval_pred):
    # eval_pred:input_id=none;label_ids=array(val-dataset-num,4096) = label;predictions=(val-dataset-num,3496,32000)=logits
    predictions, labels = eval_pred
    pad_token_id = tokenizer.pad_token_id
    predictions = np.where(predictions == -100, pad_token_id, predictions)
    # 将预测和标签转换为字符串
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 从 pred_str 提取 "<>" 之间的整数，并丢弃 preds_score 中为 None 的项及对应的 label
    filtered_results = [(extract_prediction_score(p), extract_prediction_score(l)) for p, l in zip(pred_str, label_str) if extract_prediction_score(p) is not None]
    preds_score, y = zip(*filtered_results) if filtered_results else ([], [])
    
    # 如果 preds_score 和 y 都不为空，计算 QWK
    if preds_score and y:
        qwk_score = cohen_kappa_score(preds_score, y,weights='quadratic')
        rank0_print("Quadratic Weighted Kappa (QWK):", qwk_score)
        return {"QWK": qwk_score}
    else:
        rank0_print("No valid data to compute QWK.")
        return {"QWK": 0.0}


def extract_prediction_trait_score(s):
    macro_structure_matches = re.findall(r'宏观结构得分：(\d+)', s)
    micro_narrative_matches = re.findall(r'微观结构得分：(\d+)', s)
    psychological_matches = re.findall(r'叙事心理描写得分：(\d+)', s)
    total_score_matches = re.findall(r'总分：(\d+)', s)

    macro_structure_score = int(macro_structure_matches[-1]) if macro_structure_matches else None
    micro_narrative_score = int(micro_narrative_matches[-1]) if micro_narrative_matches else None
    psychological_score = int(psychological_matches[-1]) if psychological_matches else None
    total_score = int(total_score_matches[-1]) if total_score_matches else None

    return {
        "宏观结构得分": macro_structure_score,
        "微观结构得分": micro_narrative_score,
        "叙事心理描写得分": psychological_score,
        "总分": total_score
    }

# 计算 QWK 的函数，针对不同的维度计算
def compute_trait_qwk(eval_pred):
    predictions, labels = eval_pred
    pad_token_id = tokenizer.pad_token_id
    predictions = np.where(predictions == -100, pad_token_id, predictions)
    # 将预测和标签转换为字符串
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 初始化保存每个维度的 QWK 分数
    qwk_scores = {}

    # 针对每个维度分别提取并计算 QWK
    for score_type in ["宏观结构得分", "微观结构得分", "叙事心理描写得分", "总分"]:
        # 提取相应维度的得分
        filtered_results = [(extract_prediction_trait_score(p).get(score_type), extract_prediction_trait_score(l).get(score_type)) 
                            for p, l in zip(pred_str, label_str) 
                            if extract_prediction_trait_score(p).get(score_type) is not None and extract_prediction_trait_score(l).get(score_type) is not None]

        if filtered_results:
            preds_score, y = zip(*filtered_results)
            qwk_score = cohen_kappa_score(preds_score, y, weights='quadratic')
            qwk_scores[score_type] = qwk_score
        else:
            qwk_scores[score_type] = 0.0  # 如果没有有效数据
    rank0_print("                   validate infomations:\n"+pred_str[-1][-100:])
    # 打印并返回每个维度的 QWK 分数
    for score_type, qwk in qwk_scores.items():
        # rank0_print(f"QWK for {score_type}: {qwk}")
        rank0_print(colored(f"QWK for {score_type}: {qwk}",'green'))
    return qwk_scores

# 计算 QWK 的函数，针对不同的维度计算
def graph_match_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 将预测和标签转换为字符串
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)


    rank0_print(colored("####pred infomations:\n"+pred_str[-1],'green'))
    rank0_print(colored("####label infomations:\n"+label_str[-1],'red'))


    shift_logits = torch.from_numpy(logits).contiguous()  # 去掉最后一个时间步
    shift_labels = torch.from_numpy(labels).contiguous()  # 从第一个时间步开始作为目标

    # 将logits展平为 [batch_size * sequence_length, vocab_size]
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX # 忽略填充部分
    )

    return {"cross_entropy": loss.item()}




def _train():
    global local_rank
    global tokenizer
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    rank0_print("我是rank：",local_rank)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    rank0_print("训练精度是：",compute_dtype)
    if "tmp" not in training_args.output_dir and os.path.exists(training_args.output_dir):
        if bool(os.listdir(training_args.output_dir)):
            print(f"{training_args.output_dir} already exists and not empty!!!!")
            return
        print(f"{training_args.output_dir} already exists!!!!")
    # 设置hidden-dim
    if data_args.pretrained_embedding_type in ['sbert', 'simteg_sbert']:
        model_args.mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["ofa_7b"]:
        model_args.mm_hidden_size = 4096  
    elif data_args.pretrained_embedding_type in ["ofa_13b"]:
        model_args.mm_hidden_size = 5120
    elif data_args.pretrained_embedding_type in ["vicuna_13b"]:
        model_args.mm_hidden_size = 2048  
    elif data_args.pretrained_embedding_type in ["ncla_7b"]:
        model_args.mm_hidden_size = 2048      
    elif data_args.pretrained_embedding_type in ["ncla_13b"]:
        model_args.mm_hidden_size = 2048      
    elif data_args.pretrained_embedding_type in ["GRACE_512"]:
        model_args.mm_hidden_size = 512   
    elif data_args.pretrained_embedding_type in ["ST_encoder"]:
        model_args.mm_hidden_size = 768   
    else:
        raise ValueError


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # 加载模型，

    model = NarGINALlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False
    model.config.graph_encoder_type =data_args.graph_encoder_type= model_args.graph_encoder_type
    model.config.graph_encoder_num_hidden = model_args.graph_encoder_num_hidden
    model.config.graph_encoder_num_heads = model_args.graph_encoder_num_heads
    model.config.graph_encoder_num_layers = model_args.graph_encoder_num_layers
    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    #lora

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    #加载tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
        
    #选择model_args.version，设置pad
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 初始化graph modules
    model.get_model().initialize_graph_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    data_args.is_multimodal = True#TODOis_multimodal是什么意思
    model.config.lora_enable = model_args.lora_enable = training_args.lora_enable
    if model.config.lora_enable == False: 
        #不训练LLM
        model.requires_grad_(False)
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.tune_graph_encoder = training_args.tune_graph_encoder = model_args.tune_graph_encoder
    if training_args.tune_graph_encoder:
        for p in model.get_model().graph_encoder.parameters():
            p.requires_grad = True


    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_graph_start_end = data_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    training_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Train Parameter: {name}, Size: {param.size()}")
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # rank0_print(f"Total trainable parameters: {total_params}")
        # 查看参与训练的参数情况
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(colored("Total model params: %.2fM" % (total / 1e6),'green'))
    rank0_print(colored(f'trainable params: {trainable} || all params: {total} || trainable%: {trainable / total * 100:.3f}','green'))
    #verify_model_dtype(model)
    # 构建数据集和数据collator
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    from transformers import AutoModel

    # 遍历模型的所有参数并查看哪些参数会参与训练
    # 训练器


    if data_args.is_trait_comment or data_args.is_trait:
        metric = compute_trait_qwk
        rank0_print("使用trait qwk to validate！！！")
    else:
        metric = compute_qwk
        rank0_print("使用total score qwk to validate！！！")
    if data_args.use_task == 'graph_match':
        metric = graph_match_metric
        rank0_print("使用graph match to validate！！！") 
    trainer = LLaGATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    compute_metrics=metric,
                    preprocess_logits_for_metrics=None if data_args.use_task == 'graph_match' else preprocess_logits_for_metrics ,
                    **data_module)
    #print(model)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    #model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir,safe_serialization=False)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    random.seed(0)
    _train()
