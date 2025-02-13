# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("./")
sys.path.append("./utils")
from ChildText.ChildText_dataset import ChildTextDataset


import pickle
from ASAP.gen_ASAP_data import ASAPDataset
import torch_geometric

# from ASAP.gen_ASAP_data import ASAPDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F

import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torchmetrics
from torchmetrics import MeanSquaredError
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torchmetrics
from sklearn.metrics import mean_squared_error
import math
from tqdm import tqdm
from torch.utils.data import Subset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=10,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="ASAP",
                    help="which dataset for training")
parser.add_argument("--num-heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=8,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=2048,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed")
parser.add_argument("--in-drop", type=float, default=0.0,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.2,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--dropout', type=float, default=0.3,
                    help="")
parser.add_argument('--input-dim', type=int, default=5120,
                    help="")
parser.add_argument('--gnn', type=str, default='GraphAttModel',
                    help="DeepHEATNet,GraphAttModel")



args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)




root = f'/disk/NarGINA/dataset/ChildText'
llm_name = 'vicuna-7b-v1.5'
load_texts = False
dataset = ChildTextDataset(load_texts=load_texts,root=root,encoder=None,llm_name=llm_name)

[dataset.data.__setattr__(k, torch.from_numpy(v).cuda()) for k, v in dataset.data if isinstance(v, np.ndarray)]
# node_embs = dataset.data.node_embs
# edge_embs = dataset.data.edge_embs

# 创建 DataLoader
#train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
train_dataset, val_dataset, test_dataset = Subset(dataset=dataset,indices=dataset.side_data['train']),Subset(dataset=dataset,indices=dataset.side_data['valid']),Subset(dataset=dataset,indices=dataset.side_data['test'])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


train_data = []
val_data = []
test_data = []



for i,g in enumerate(train_loader):
    if g.edge_index.size(0) == 0 or g.edge_index.size(1) == 0:
        print("邻接矩阵空")
        continue
    if g.edge_index.min() < 0:
        print("邻接矩阵存在负数")
        continue
    g.to("cuda")
    data = g.to_data_list()[0]
    data.essay_text = data.source_text
    del data.source_text
    train_data.append(data)


for i,g in enumerate(val_loader):
    if g.edge_index.size(0) == 0 or g.edge_index.size(1) == 0:
        print("邻接矩阵空")
        continue
    if g.edge_index.min() < 0:
        print("邻接矩阵存在负数")
        continue
    g.to("cuda")
    data = g.to_data_list()[0]
    data.essay_text = data.source_text
    del data.source_text
    val_data.append(data)



for i,g in enumerate(test_loader):

    if g.edge_index.size(0) == 0 or g.edge_index.size(1) == 0:
        print("邻接矩阵空")
        continue
    if g.edge_index.min() < 0:
        print("邻接矩阵存在负数")
        continue
    data = g.to_data_list()[0]
    data.essay_text = data.source_text
    del data.source_text
    test_data.append(data)



print("处理完毕")
#print(all_data)
with open('/disk/NarGINA/dataset/ChildText/embedings/vicuna-7b-v1.5/pretrained_ChildText_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('/disk/NarGINA/dataset/ChildText/embedings/vicuna-7b-v1.5/pretrained_ChildText_val.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('/disk/NarGINA/dataset/ChildText/embedings/vicuna-7b-v1.5/pretrained_ChildText_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)