import os


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import pickle
import sys
import numpy as np
import torch
sys.path.append("./")
sys.path.append("./ChildText")
from torch_geometric.data import DataLoader
from ChildText.ChildText_dataset import ChildTextDataset
from torch.utils.data import Subset
import torch



def get_embeds(data_loader,embs_path,hidden_dim,model_name,mode):
    assert data_loader.batch_size == 1 ,"get embs:batchsize must set 1."
    os.makedirs(embs_path,exist_ok=True)
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
            data = batch_data.to_data_list()[0]
            data.essay_text = data.source_text
            del data.source_text
            data_list.append(data)
    save_path = embs_path + f'/pretrained_{model_name}_hidden={hidden_dim}_{mode}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Save : {save_path}; length={len(data_list)}")



def main():
    device = torch.device('cuda')
    embs_path = '/disk/NarGINA/dataset/ChildText_test/embedings/text_encoder'
    model_name = 'text-encoder'


    root = f'/disk/NarGINA/dataset/ChildText_test'
    llm_name = 'ST_ofa_768'#vicuna-7b-v1.5
    load_texts = False
    dataset = ChildTextDataset(load_texts=load_texts,root=root,encoder=None,llm_name=llm_name)
    [dataset.data.__setattr__(k, torch.from_numpy(v).cuda()) for k, v in dataset.data if isinstance(v, np.ndarray)]
    hidden_dim = dataset.num_features

    train_dataset, val_dataset,test_dataset = Subset(dataset=dataset,indices= dataset.side_data['train'] ),Subset(dataset=dataset,indices=dataset.side_data['valid']),Subset(dataset=dataset,indices=dataset.side_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    

    get_embeds(data_loader=train_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='train')
    get_embeds(data_loader=val_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='val')
    get_embeds(data_loader=test_loader,embs_path=embs_path,hidden_dim=hidden_dim,model_name=model_name,mode='test')


if __name__ == '__main__':
    main()