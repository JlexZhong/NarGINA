import os
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
# class GConv(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, activation, num_layers):
#         super(GConv, self).__init__()
#         self.activation = activation()
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

#     def forward(self, x, edge_index, edge_weight=None):
#         z = x
#         for i, conv in enumerate(self.layers):
#             z = conv(z, edge_index, edge_weight)
#             z = self.activation(z)
#         return z
# class RGAT(torch.nn.Module): 
#     def __init__(self, input_dim, hidden_dim, activation, num_layers, heads=1):
#         super(RGAT, self).__init__()
#         self.activation = activation()
#         self.layers = torch.nn.ModuleList()

#         # 输入层
#         self.layers.append(RGATConv(input_dim, hidden_dim, heads=heads, edge_dim= input_dim, concat=True))

#         # 隐藏层
#         for _ in range(num_layers - 2):
#             self.layers.append(RGATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim= input_dim, concat=True))

#         # 输出层，不使用concat，将heads合并为输出维度
#         self.layers.append(RGATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim= input_dim, concat=False))

#     def forward(self, x, edge_index, edge_attr=None,edge_type = None):
#         z = x
#         for i, conv in enumerate(self.layers):
#             # 在前向传播中传递 edge_attr
#             z = conv(z, edge_index, edge_attr=edge_attr,edge_type=edge_type)
#             z = self.activation(z)
#         return z


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers, heads=1):
        super(GAT, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)  # 使用 LeakyReLU 代替 ReLU
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()  # 添加 BatchNorm 列表

        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        self.batch_norms.append(BatchNorm(hidden_dim * heads))  # 第一个 BatchNorm

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.batch_norms.append(BatchNorm(hidden_dim * heads))  # 隐藏层 BatchNorm

        # 输出层，不使用concat，将heads合并为输出维度
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        self.batch_norms.append(BatchNorm(hidden_dim))  # 最后一层 BatchNorm

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, gat_layer in enumerate(self.layers):
            z = gat_layer(z, edge_index, edge_attr=edge_weight)
            z = self.batch_norms[i](z)  # 应用 BatchNorm
            z = self.activation(z)
            z = self.dropout(z)  # 应用 Dropout
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
class GraphScoreModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes=21):
        super(GraphScoreModel, self).__init__()

        # 多层全连接网络
        self.fc1 = Linear(hidden_dim, hidden_dim // 4)
        self.fc2 = Linear(hidden_dim // 4, num_classes)

        # 激活函数和dropout（可选）
        self.relu = ReLU()

    def forward(self, z, batch):
        # 使用torch_geometric的全局平均池化操作，得到图级别的表示
        # z: (num_nodes, hidden_dim), batch: (num_nodes,)
        graph_embedding = global_mean_pool(z, batch)  # 对每个图的节点特征取平均

        # 前向传播，经过多层全连接网络
        x = self.fc1(graph_embedding)
        x = self.relu(x)
        
        out = self.fc2(x)  # 最后输出11个类别
        return out


def train(encoder_model, contrast_model, train_loader, optimizer,epoch_num):
    with tqdm(total=epoch_num, desc='(对比学习预训练)') as pbar:
        encoder_model.train()
        for epoch in range(1, epoch_num+1):
            loss_sum = 0.0
            steps = 0
            for i,batch_data in enumerate(train_loader):
                batch_data.to("cuda")
                optimizer.zero_grad()
                z, z1, z2 = encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
                h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
                loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()
                steps+=1
            pbar.set_postfix({'loss': loss_sum/steps})
            pbar.update()
            print(f"Epoch {epoch}, loss={loss_sum/steps:.3f}")
    print("对比学习-----训练结束")
    return loss_sum/steps


def Task_train(encoder_model, score_model,train_loader,optimizer,epoch_num):
    loss_fct = torch.nn.CrossEntropyLoss()
    with tqdm(total=epoch_num, desc='(score essay task)') as pbar:
        encoder_model.eval()
        for epoch in range(1, epoch_num+1):
            loss_sum = 0.0
            steps = 0
            for i,batch_data in enumerate(train_loader):
                batch_data.to("cuda")
                optimizer.zero_grad()

                # 获取编码器的图嵌入
                z, _, _ = encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)

                # 使用评分模型预测分数
                preds_score = score_model(z,batch_data.batch)

                # 计算损失
                loss = loss_fct(preds_score, batch_data.y.squeeze())  # 注意，y应为1D分类标签
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                steps += 1
            pbar.set_postfix({'loss': loss_sum/steps})
            pbar.update()
            print(f"Task Epoch {epoch}, loss={loss_sum/steps:.3f}")
    print("下游任务-----训练结束")

    return loss_sum/steps

def test(encoder_model, score_model, test_loader):
    encoder_model.eval()
    score_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data.to("cuda")
            z, _, _ = encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
            preds_score = score_model(z,batch_data.batch)

            # 取预测分数最高的类
            _, predicted = torch.max(preds_score, dim=1)
            all_preds.extend(predicted.cpu().numpy())  # 保存预测标签
            all_labels.extend(batch_data.y.view(1).cpu().numpy())  # 保存真实标签

    # 计算 QWK
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    print(all_preds)
    print(f'Quadratic Weighted Kappa (QWK): {qwk:.3f}')
    return qwk


def get_embeds(encoder_model, score_model, data_loader):
    assert data_loader.batch_size == 1 ,"get embs:batchsize must set"
    encoder_model.eval()
    score_model.eval()
    data_list = []

    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data.edge_index.size(0) == 0 or batch_data.edge_index.size(1) == 0:
                print("邻接矩阵空")
                continue
            if batch_data.edge_index.min() < 0:
                print("邻接矩阵存在负数")
                continue
            batch_data.to("cuda")
            z, _, _ = encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
            data = batch_data.to_data_list()[0]
            data.essay_text = data.source_text
            data.x = z
            del data.source_text
            data_list.append(data)

    return data_list



def main():
    device = torch.device('cuda')
    weights_path = '/disk/NarGINA/contrastive_learning/weights/ChildText/sum_score'
    epoch_num = 200
    task_epoch_num = 150
    heads_num = 8
    num_layers = 4
    hidden_dim = 512
    proj_dim = 512
    num_classes = 21
    root = f'/disk/NarGINA/dataset/ChildText/sum_score'
    llm_name = 'ST_ofa_768'#vicuna-13b-v1.5
    load_texts = False
    dataset = ChildTextDataset(load_texts=load_texts,root=root,encoder=None,llm_name=llm_name)
    [dataset.data.__setattr__(k, torch.from_numpy(v).cuda()) for k, v in dataset.data if isinstance(v, np.ndarray)]

    train_dataset, test_dataset = Subset(dataset=dataset,indices= dataset.side_data['train'] + dataset.side_data['valid']),Subset(dataset=dataset,indices=dataset.side_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gat = GAT(input_dim=dataset.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=num_layers,heads=heads_num).to(device)
    encoder_model = Encoder(encoder=gat, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim = proj_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    print(encoder_model)

    pretrain_loss = train(encoder_model, contrast_model, train_loader, optimizer,epoch_num)
    #pretrain_loss = 3.92
    torch.save(encoder_model.state_dict(), weights_path+'/'+f"encoder_model_{llm_name}_hidden={hidden_dim}_numlayers={num_layers}_headnum={heads_num}_loss={pretrain_loss:.2f}_weights.pth")
    print("Encoder model weights saved.")

    encoder_model.load_state_dict(torch.load(weights_path+'/'+f"encoder_model_{llm_name}_hidden={hidden_dim}_numlayers={num_layers}_headnum={heads_num}_loss={pretrain_loss:.2f}_weights.pth"))
    print("Encoder model weights loaded.")

    score_model =  GraphScoreModel(hidden_dim=hidden_dim,num_classes=num_classes).to(device)  # 请你设计一个作文评分模型
    optimizer = Adam(score_model.parameters(), lr=0.01)
    
    task_loss = Task_train(encoder_model,score_model,train_loader,optimizer,task_epoch_num)
    qwk = test(encoder_model, score_model, test_loader)  # 测试模型
    torch.save(score_model.state_dict(),weights_path+'/'+f"score_model_{llm_name}_hidden={hidden_dim}_numlayers={num_layers}_headnum={heads_num}_qwk={qwk:.2f}_weights.pth")
    print("score model weights saved.")
    # score_model.load_state_dict(torch.load(weights_path+'/'+f"score_model_{llm_name}_hidden={hidden_dim}_weights.pth"))
    # print("score_model weights loaded.")



if __name__ == '__main__':
    main()