import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import time
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from torch import Tensor
from torch import nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from transformers import BitsAndBytesConfig
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel,AutoModelForCausalLM)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

import json
import os
import os.path as osp
import random
import time
from datetime import datetime
from itertools import product
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from sklearn.model_selection import StratifiedKFold


import inspect
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import HEATConv, BatchNorm


LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120,"bert-base-chinese":768,"llama2_7b_Chinese":4096,"llama2_7b_hf":4096,"vicuna-13b-v1.5":5120,"vicuna-7b-v1.5":4096}

def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()

def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def resolver(
    classes: List[Any],
    class_dict: Dict[str, Any],
    query: Union[Any, str],
    base_cls: Optional[Any],
    base_cls_repr: Optional[str],
    *args,
    **kwargs,
):

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ""
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


def activation_resolver(query: Union[Any, str] = "relu", *args, **kwargs):
    import torch

    base_cls = torch.nn.Module
    base_cls_repr = "Act"
    acts = [
        act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(
        acts, act_dict, query, base_cls, base_cls_repr, *args, **kwargs
    )

def load_pretrained_state(model_dir, deepspeed=False):
    if deepspeed:
        def _remove_prefix(key: str, prefix: str) -> str:
            return key[len(prefix):] if key.startswith(prefix) else key
        state_dict = get_fp32_state_dict_from_zero_checkpoint(model_dir)
        state_dict = {_remove_prefix(k, "_forward_module."): state_dict[k] for k in state_dict}
    else:
        state_dict = torch.load(model_dir)["state_dict"]
    return state_dict


class MLP(torch.nn.Module):
    """
    MLP model modifed from pytorch geometric.
    """

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        dropout: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: bool = True,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {})) # 激活函数
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.0
        elif len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list)-1})"
            )
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})"
            )

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(
                torch.nn.Linear(in_channels, out_channels, bias=_bias) # 768,1536
            )
        # [768,1536]=>[1536,768]=>[768,1]
        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = torch.nn.BatchNorm1d(hidden_channels)
            else:
                norm_layer = torch.nn.Identity()
            self.norms.append(norm_layer)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of to the
                final output layer. (default: :obj:`False`)
        """
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)

        if self.plain_last: # 全拉成一维
            x = self.lins[-1](x) # [33,1]
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"


class SmartTimer:
    """A timer utility that output the elapsed time between this
    call and last call.
    """

    def __init__(self, verb=True) -> None:
        """SmartTimer Constructor

        Keyword Arguments:
            verb {bool} -- Controls printing of the timer (default: {True})
        """
        self.last = time.time()
        self.verb = verb

    def record(self):
        """Record current timestamp"""
        self.last = time.time()

    def cal_and_update(self, name):
        """Record current timestamp and print out time elapsed from last
        recorded time.

        Arguments:
            name {string} -- identifier of the printout.
        """
        now = time.time()
        if self.verb:
            print(name, now - self.last)
        self.record()
    
class MultiLayerMessagePassing(nn.Module, metaclass=ABCMeta):
    """Message passing GNN"""

    def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            drop_ratio=None,
            JK="last",
            batch_norm=True,
    ):
        """

        :param num_layers: layer number of GNN
        :type num_layers: int
        :param inp_dim: input feature dimension
        :type inp_dim: int
        :param out_dim: output dimension
        :type out_dim: int
        :param drop_ratio: layer-wise node dropout ratio, defaults to None
        :type drop_ratio: float, optional
        :param JK: jumping knowledge, should either be ["last","sum"],
        defaults to "last"
        :type JK: str, optional
        :param batch_norm: Use node embedding batch normalization, defaults
        to True
        :type batch_norm: bool, optional
        """
        super().__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.drop_ratio = drop_ratio

        self.conv = torch.nn.ModuleList()

        if batch_norm:
            self.batch_norm = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))
        else:
            self.batch_norm = None

        self.timer = SmartTimer(False)

    def build_layers(self):
        for layer in range(self.num_layers):
            if layer == 0:
                self.conv.append(self.build_input_layer())
            else:
                self.conv.append(self.build_hidden_layer())

    @abstractmethod
    def build_input_layer(self):
        pass

    @abstractmethod
    def build_hidden_layer(self):
        pass

    @abstractmethod
    def layer_forward(self, layer, message):
        pass

    @abstractmethod
    def build_message_from_input(self, g):
        pass

    @abstractmethod
    def build_message_from_output(self, g, output):
        pass

    def forward(self, g, drop_mask=None):
        h_list = []
        dtype = g.x.dtype
        message = self.build_message_from_input(g)

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message).to(dtype)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                dropped_h = F.dropout(h, p=self.drop_ratio, training=self.training)
                if drop_mask is not None:
                    h = drop_mask.view(-1, 1) * dropped_h + torch.logical_not(drop_mask).view(-1, 1) * h
                else:
                    h = dropped_h
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        elif self.JK == "mean":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
            repr = repr/self.num_layers
        else:
            repr = h_list
        return repr

def masked_edge_index(edge_index, edge_mask):
    #print(edge_mask)#TODO
    if isinstance(edge_index, torch.Tensor):
        return edge_index[:, edge_mask]

class RGCNEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(
            torch.empty(self.num_relations, in_channels, out_channels)
        )

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        #message["h"], message["he"], message["g"], message["e"]
        # #{
        #     "g": g.edge_index,
        #     "h": g.x,
        #     "e": g.edge_type,#FIXME edge_type改成了edge_types
        #     "he": g.edge_attr,
        # }
        out = torch.zeros(x.size(0), self.out_channels, device=x.device,dtype=x.dtype) # [292,768]
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask) # [2,284] 获得mask后的edge index
            t = xe[edge_mask]
            h = self.propagate(tmp, x=x, xe=xe[edge_mask]).to(x.dtype) # 292,768
            out += h @ self.weight[i]# [5,768,768]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]
        #都是[xxx,768]
        # Step 4: Normalize node features.
        return (x_j + xe).relu()

class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int, # 5
        inp_dim: int, # 768
        out_dim: int, # 768
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, x,edge_index,edge_attr,edge_type):
        return {
            "g": edge_index,
            "h": x,
            "e": edge_type,#FIXME edge_type改成了edge_types
            "he": edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}#TODO

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )


class SingleHeadAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # 定义 W_q, W_k, W_v 分别用于 query, key, value 的线性变换
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        #self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        #[node-num,num_layers,768]
        batch_size = query.size(0)

        # 线性变换，得到 [node-num,num_layers, embed_dim]
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # 将 Q, K, V reshape 为 [node-num, num_heads, num_layers, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_layer, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_layer, head_dim]

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, 1, num_layer]

        # 计算每个头的注意力输出
        context = torch.matmul(attn_weights, V).squeeze(dim=2)  # [batch_size, num_heads, 1, head_dim]

        # # 将 heads 组合回去
        attn_output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim).squeeze()  # [batch_size, 1, embed_dim]
        # # 输出线性变换
        # output = self.out_proj(context)
        return context,attn_output


class attn_fusion(torch.nn.Module):
    def __init__(self, attn_out_dim,**kwargs):
        super(attn_fusion).__init__()
        self.out_proj = nn.Linear(attn_out_dim, attn_out_dim)

    def forward(self,attn_output):
        # # 输出线性变换
        output = self.out_proj(attn_output)
        return output



class GraphAttModel(torch.nn.Module):
    """
    GNN model that use a single layer attention to pool final node representation across
    layers.
    """
    def __init__(self, llm_name, outdim, add_rwpe=None, num_heads=8,gnn_in_dim=None,gnn_out_dim=None,gnn_num_layers=None,**kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        # outdim=4096

        self.model = PyGRGCNEdge(
                        num_layers=gnn_num_layers,
                        num_rels=5, # num_relations
                        inp_dim=gnn_in_dim,
                        out_dim=gnn_out_dim,
                        drop_ratio=0.2,
                        JK='none',
            )
        
        self.llm_name = llm_name
        self.outdim = outdim
        self.num_heads = num_heads
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], gnn_in_dim)

        self.att = MultiHeadAttention(embed_dim = outdim,num_heads=num_heads)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None
        self.out_proj = nn.Linear(outdim, outdim)

    def initial_projection(self, x,edge_attr):
        x = self.llm_proj(x)
        edge_attr = self.llm_proj(edge_attr)
        return x,edge_attr

    def forward(self, x,edge_index,edge_attr,edge_type,batch=None):
        #TODO 把batchsize设为1
        x,edge_attr = self.initial_projection(x,edge_attr)

        emb = torch.stack(self.model(x=x,edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type), dim=1) # [node-num,num_layers,768]
        query = x.unsqueeze(1) # [node-num,1,768]
        heads,attn_output = self.att(query,emb, emb) 

        node_embs = self.out_proj(attn_output)

        return node_embs

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False


# 定义一个更深层次的HEATNet模型
class DeepHEATNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 proj_channels,
                 hidden_channels, 
                 out_channels, 
                 num_node_types, 
                 num_edge_types, 
                 edge_type_emb_dim, 
                 edge_dim, 
                 edge_attr_emb_dim, 
                 heads, 
                 num_layers, 
                 dropout):
        super(DeepHEATNet, self).__init__()
        self.head_num = heads
        self.out_channels = out_channels
        # 初始化多层HEATConv和批量归一化层
        self.convs = ModuleList()
        self.bns = ModuleList()
        
        self.llm_proj = nn.Linear(in_channels, proj_channels)

        # 第一层：从输入通道转换为隐藏通道
        self.convs.append(HEATConv(proj_channels, hidden_channels, num_node_types, 
                                   num_edge_types, edge_type_emb_dim, edge_dim, 
                                   edge_attr_emb_dim, heads))
        
        self.bns.append(BatchNorm(hidden_channels * heads))
        
        # 中间层：堆叠多层HEATConv，使用隐藏通道
        for _ in range(num_layers - 2):
            self.convs.append(HEATConv(hidden_channels * heads, hidden_channels, 
                                       num_node_types, num_edge_types, 
                                       edge_type_emb_dim, edge_dim, edge_attr_emb_dim, heads))
            self.bns.append(BatchNorm(hidden_channels * heads))
        
        # 最后一层：从隐藏通道转换为输出通道
        self.convs.append(HEATConv(hidden_channels * heads, out_channels, 
                                   num_node_types, num_edge_types, edge_type_emb_dim, 
                                   edge_dim, edge_attr_emb_dim, heads=heads))  # 输出不需要多个heads
        self.dropout = dropout


    def initial_projection(self, x,edge_attr):
        # 降维 
        x = self.llm_proj(x)
        edge_attr = self.llm_proj(edge_attr)
        return x,edge_attr
    
    def forward(self, batch_data):
        node_type = torch.zeros(batch_data.x.size(0), dtype=torch.long).cuda()
        x,edge_index,edge_type,edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_types, batch_data.edge_attr
        if (edge_type < 0).any():
            pos =  edge_type >= 0
            edge_type = edge_type[pos]
            edge_index = edge_index[:,pos]
            edge_attr = edge_attr[pos]
        x,edge_attr = self.initial_projection(x,edge_attr)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, node_type, edge_type, edge_attr)
            x = self.bns[i](x)  # 批量归一化
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout 防止过拟合
            
        # 最后一层不需要激活函数和dropout
        x = self.convs[-1](x, edge_index, node_type, edge_type, edge_attr)

        heads = x.contiguous().view(-1,self.head_num,self.out_channels)

        return heads
    

