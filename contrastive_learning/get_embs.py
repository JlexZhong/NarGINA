import os
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import sys
import numpy as np
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
sys.path.append("./")
sys.path.append("./ChildText")
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv,GATConv,RGATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from ChildText.ChildText_dataset import ChildTextDataset
from torch.utils.data import Subset
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import cohen_kappa_score
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, BatchNorm
import torch
import torch.nn as nn

from contrastive_learning.GRACE import GAT, Encoder, GraphScoreModel, Task_train, train


def get_embeds(encoder_model, data_loader,embs_path,hidden_dim,model_name,mode):
    assert data_loader.batch_size == 1 ,"get embs:batchsize must set 1."
    os.makedirs(embs_path,exist_ok=True)
    encoder_model.eval()
    data_list = []

    with torch.no_grad():
        for batch_data in data_loader:
            # if batch_data.edge_index.size(0) == 0 or batch_data.edge_index.size(1) == 0:
            #     print("邻接矩阵空")
            #     continue
            # if batch_data.edge_index.min() < 0:
            #     print("邻接矩阵存在负数")
            #     continue
            batch_data.to("cuda")
            z, _, _ = encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
            data = batch_data.to_data_list()[0]
            data.essay_text = data.source_text
            data.x = z
            del data.source_text
            data_list.append(data)
    save_path = embs_path + f'/pretrained_{model_name}_hidden={hidden_dim}_{mode}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Save : {save_path}; length={len(data_list)}")



def main():
    device = torch.device('cuda')
    embs_path = '/disk/NarGINA/dataset/ChildText_test/teacher_2/embedings/GRACE_ST_t5uie-vicuna'
    model_name = 'GAT'

    encoder_weight_path = "/disk/NarGINA/contrastive_learning/weights/ChildText/v1/encoder_model_ST_hidden=512_numlayers=3_headnum=4_loss=3.89_weights.pth"
    # 使用正则表达式匹配参数及其值
    parameters = {
        "hidden": re.search(r"hidden=(\d+)", encoder_weight_path).group(1),
        "numlayers": re.search(r"numlayers=(\d+)", encoder_weight_path).group(1),
        "headnum": re.search(r"headnum=(\d+)", encoder_weight_path).group(1),
        "loss": re.search(r"loss=([\d.]+)", encoder_weight_path).group(1),
    }
    heads_num = int(parameters["headnum"]) 
    num_layers = int(parameters["numlayers"]) 
    hidden_dim = int(parameters["hidden"]) 
    proj_dim = int(parameters["hidden"]) 

    root = f'/disk/NarGINA/dataset/ChildText_test/teacher_2/t5uie-vicuna_graph'
    llm_name = 'ST_ofa_768'#vicuna-7b-v1.5
    load_texts = False
    dataset = ChildTextDataset(load_texts=load_texts,root=root,encoder=None,llm_name=llm_name)
    [dataset.data.__setattr__(k, torch.from_numpy(v).cuda()) for k, v in dataset.data if isinstance(v, np.ndarray)]

    train_dataset, val_dataset,test_dataset = Subset(dataset=dataset,indices= dataset.side_data['train'] ),Subset(dataset=dataset,indices=dataset.side_data['valid']),Subset(dataset=dataset,indices=dataset.side_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    gat = GAT(input_dim=dataset.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=num_layers,heads=heads_num).to(device)
    encoder_model = Encoder(encoder=gat, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim = proj_dim).to(device)
    print(encoder_model)
    encoder_model.load_state_dict(torch.load(encoder_weight_path))
    print("Encoder model weights loaded.")

    get_embeds(encoder_model=encoder_model,data_loader=train_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='train')
    get_embeds(encoder_model=encoder_model,data_loader=val_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='val')
    get_embeds(encoder_model=encoder_model,data_loader=test_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='test')


if __name__ == '__main__':
    main()