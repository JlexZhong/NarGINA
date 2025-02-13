import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import sys
sys.path.append('./')
sys.path.append('./utils')
from utils.encode_graph import OFAPygDataset, SentenceEncoder

import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg

from ASAP.gen_ASAP_data import ASAPDataset


EDGE_TYPE_MAPPING = {
    "使能-因果": 0,
    "动机-因果": 1,
    "物理-因果": 2,
    "心理-因果": 3,
    "并列": 4
}

def load_source_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# 从 name.nodes.json 中提取节点文本 X
def load_node_Text(nodes_file):
    data = []
    with open(nodes_file, 'r', encoding='utf-8') as f:
        for j in f.readlines():
            # 将josn字符串转化为dict字典
            j = json.loads(j)
            data.append(j)
    X = [node['transformed_text'] for node in data]
    return X

# 从 name.edges.json 中提取邻接矩阵 edge_index 和边特征 edge_feature
def load_edge_data(edges_file):
    data = []
    with open(edges_file, 'r', encoding='utf-8') as f:
        for j in f.readlines():
            # 将josn字符串转化为dict字典
            j = json.loads(j)
            data.append(j)
    edge_index = []
    edge_feature = []
    for edge in data:
        if 'first_index' in edge and 'second_index' in edge:
            if EDGE_TYPE_MAPPING.get(edge['relation_type'], -1) != -1:
                edge_index.append([edge['first_index'], edge['second_index']])
                edge_feature.append(edge['relation_type'])
        else:
            # print(edges_file,'index:',edge['index'],'error')
            continue#TODO存在某些标注错误，导致没有提取到对应的first index
    return torch.from_numpy(np.array(edge_index).T).long(), edge_feature

def load_my_dataset(raw_data_path):
    # Read data into huge `Data` list.
    graphs = []
    with open(os.path.join(raw_data_path,'split_index.json'), 'r', encoding='utf-8') as file:
        split_index = json.load(file)   
    with open(os.path.join(raw_data_path,'updated_total_data_sum.json'), 'r', encoding='utf-8') as file:
        reader = json.load(file)   
        for id , row in enumerate(reader): 
            filename = row['filename']
            nodes_file = os.path.join(raw_data_path,'raw',filename+'.nodes.json')
            edges_file = os.path.join(raw_data_path,'raw',filename+'.edges.json')
            if not os.path.exists(nodes_file) :
                if not os.path.exists(edges_file) :
                    print(nodes_file,'not exist!')
                    continue
            source_text = row['essay_text']
            comment = row['comment'] if row['comment'] is not None else -1
            macro_score = row['macro_score'] if row['macro_score'] is not None else -1
            micro_score = row['micro_score'] if row['micro_score'] is not None else -1
            psych_score = row['psych_score'] if row['psych_score'] is not None else -1
            total_score = row['total_score']

            node_text = load_node_Text(nodes_file)
            edge_index, edge_text = load_edge_data(edges_file)

            if edge_index.size(0) == 0 or edge_index.size(1) <= 1 or edge_index.max() >=len(node_text):
                print("错误数据")
                continue

            graph = dict()
            graph["edge_list"] = edge_index
            graph["edge_feat"] = edge_text
            graph['edge_type'] = torch.tensor([EDGE_TYPE_MAPPING[edge] for edge in edge_text ]) 
            graph["node_feat"] = node_text
            graph["comment"] = comment
            graph["source_text"] = source_text
            graph["macro_score"] = macro_score
            graph["micro_score"] = micro_score
            graph["psych_score"] = psych_score
            graph["total_score"] = total_score
            y = int(total_score)
            graph['label'] = np.array(y) 

            if id in split_index['train']:
                graph['split'] = 'train'
            elif id in split_index['valid']:
                graph['split'] = 'valid'
            elif id in split_index['test']:
                graph['split'] = 'test'
            else:
                graph['split'] = None
                print("找不到划分index")
            graphs.append(graph)

    print("gen graph")
    node_texts = []
    edge_texts = []
    data = []
    comment_text=[]
    essay_text = []
    for g in graphs:
        node_texts += g["node_feat"] # len=18132
        edge_texts += g["edge_feat"]
    unique_node_texts = set(node_texts)
    unique_edge_texts = set(edge_texts)
    u_node_texts_lst = list(unique_node_texts)
    u_edge_texts_lst = list(unique_edge_texts)
    node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)} # len=11365,给每一种节点事件，赋予一个id，去掉了重复
    edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}

    split = {"train": [], "valid": [], "test": []}
    for i, g in enumerate(graphs):
        cur_nt_id = [node_texts2id[v] for v in g["node_feat"]] # 节点的id的list
        cur_et_id = [edge_texts2id[v] for v in g["edge_feat"]]
        if len(cur_nt_id) <= 1:
            print("错误文件！！！")
            continue
        data.append(pyg.data.Data(x=torch.tensor(cur_nt_id, dtype=torch.long), # 每个事件的特征是id，而不是embedding
                                        xe=torch.tensor(cur_et_id, dtype=torch.long),
                                        edge_index=g["edge_list"],
                                        edge_type=g['edge_type'],
                                        y=torch.tensor(g["label"]), 
                                        comment= g['comment'],
                                        source_text=g['source_text'],
                                        macro_score = torch.tensor(g['macro_score']),
                                        micro_score = torch.tensor(g['micro_score']),
                                        psych_score = torch.tensor(g['psych_score']),
                                        total_score = torch.tensor(g['total_score']),# string
                                        ))
        comment_text.append(g['comment'])
        essay_text.append(g['source_text'])
        split[g["split"]].append(i)

    prompt_edge_text = ["prompt edge.", 
                        "prompt edge. This is a children's narrative text, and the narrative graph is extracted from this narrative text. When rating, it is necessary to combine narrative text and narrative graph.",
                        ]  
    #prompt_text = ["prompt node. The target task is a graph regression task, which assigns a score level to the children's narrative text represented by the narrative graph, mainly focusing on the completeness and coherence of the narrative. The score must be between 0.0 and 10.0 points. A narrative diagram is a graph extracted from unstructured children's narrative text, where nodes represent an event, and the directed edges between nodes represent the relationships between events. The theme of all narrative graph is \"Frog, Where Are You?\".", # TODO
    #                 "prompt node. The target task is a graph regression task, which assigns a score level to the children's narrative text represented by the narrative graph, mainly focusing on the completeness and coherence of the narrative. The score must be between 0.0 and 10.0 points. A narrative diagram is a graph extracted from unstructured children's narrative text, where nodes represent an event, and the directed edges between nodes represent the relationships between events. The theme of all narrative graph is \"Frog, Where Are You?\".", ]
    #labels_features = ["prompt node. The target task is a graph regression task, which assigns a score level to the children's narrative text represented by the narrative graph, mainly focusing on the completeness and coherence of the narrative. The score must be between 0.0 and 10.0 points. A narrative diagram is a graph extracted from unstructured children's narrative text, where nodes represent an event, and the directed edges between nodes represent the relationships between events. The theme of all narrative graph is \"Frog, Where Are You?\".",]
    prompt_text = ["prompt node. The target task is a graph score task. The input of the model is a child's narrative text and corresponding narrative graph. The model assigns a score level to the child's narrative text represented by the narrative graph, mainly focusing on event completeness, causal coherence, and lexical and syntactic complexity. The rating categories range from 0 to 10, with a total of 10 categories. A narrative graph is a graph extracted from unstructured children's narrative text, where nodes represent events and directed edges between nodes represent relationships between events (parallel relationships; motivational, psychological, physical, and enabling causal relationships). The theme of all narrative graph is' Frog, where are you? '.", ]


    ret = (data, 
           [u_node_texts_lst, u_edge_texts_lst, prompt_edge_text, prompt_text],#TODO添加了essay text
           split)
    return ret


class ChildTextDataset(OFAPygDataset):
    
    def gen_data(self):
        raw_data_path = '/disk/NarGINA/ChildText/raw_graph'
        pyg_graph, texts, split = load_my_dataset(raw_data_path)
        return [d for d in pyg_graph], texts, split


    def add_raw_texts(self, data_list, texts):
        data, slices = self.collate(data_list)
        data.node_embs = np.array(texts[0])
        data.edge_embs = np.array(texts[1])
        data.prompt_edge_text_feat = np.array(texts[2])
        data.noi_node_text_feat = np.array(texts[3])
        return data, slices


    def add_text_emb(self, data_list, text_emb):
        """
        This function assigns generated embedding to member variables of the graph

        data_list: data list returned in self.gen_data.
        text_emb: list of torch text tensor corresponding to the returned text in self.gen_data. text_emb[0] = llm_encode(text[0])

        Since the majority of node/edge text embeddings are repeated, we only store unique
        ones, and keep the indices.
        """
        data, slices = self.collate(data_list)#Data(x=[19916], edge_index=[2, 16027], y=[543], xe=[16027])
        data.node_embs = text_emb[0]
        data.edge_embs = text_emb[1]
        # data.prompt_edge_text_feat = text_emb[2]
        # data.noi_node_text_feat = text_emb[3]
        # data.essay_text = text_emb[5]# TODO

        return data, slices

    def get(self, index):
        # dataset[index]
        data = super().get(index)
        node_feat = self.node_embs[data.x.numpy()]
        edge_feat = self.edge_embs[data.xe.numpy()]
        data.node_text =  [self.text[0][i] for i in data.x.tolist()]
        data.x = node_feat
        data.edge_attr = edge_feat
        data.y = data.y.view(1, -1)

        return data

    def get_idx_split(self):
        """
        Return the split information required to split the dataset, this optional, you can further split the dataset in task_constructor.py
        
        """
        return self.side_data[0]

    def get_task_map(self):
        """
        Because a dataset can have multiple different tasks that requires different prompt/class text embedding. This function returns a task map that maps a task name to the desired text embedding. Specifically, a task map is of the following format.

        prompt_text_map = {task_name1: {"noi_node_text_feat": ["noi_node_text_feat", [$Index in data[0].noi_node_text_feat$]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             [$Index in data[0].class_node_text_feat$]],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [$Index in data[0].prompt_edge_text_feat$]]},
                       task_name2: similar to task_name 1}
        Please refer to examples in data/ for details.
        """
        return self.side_data[1]

    def get_edge_list(self, mode="e2e",is_essay=True):
        """
        Defines how to construct prompt graph
        f2n: noi nodes to noi prompt node
        n2f: noi prompt node to noi nodes
        n2c: noi prompt node to class nodes
        c2n: class nodes to noi prompt node
        For different task/mode you might want to use different prompt graph construction, you can do so by returning a dictionary. For example
        {"f2n":[1,0], "n2c":[2,0]} means you only want f2n and n2c edges, f2n edges have edge type 1, and its text embedding feature is data[0].prompt_edge_text_feat[0]
        """
        if mode == "e2e_graph":
            if is_essay:
                return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]],"essay_2_noi":[4,[1]]}#TODO回归任务中考虑添加c2n
            else:
                return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]]}#
        #[1,[0]]第一个位置指示边的类型，第二个位置[0]用于索引prompt_edge_text_feat中的txt-emb，这条边的emb=prompt_edge_text_feat【0】
            #return {"f2n": [1, [0]], "n2f": [2, [0]], "n2c": [3, [0]]}

        elif mode == "lr_graph":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}


class ChildText_Dataset_Aug(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None,node_embs=None,edge_embs=None):
        """
        :param root: 数据集保存的根目录
        :param data_list: 图数据列表，每个元素为 torch_geometric.data.Data 对象
        :param transform: 可选的变换函数
        :param pre_transform: 可选的预处理函数，在保存之前执行
        """
        self.data_list = data_list
        self.node_embs = node_embs
        self.edge_embs = edge_embs
        super(ChildText_Dataset_Aug, self).__init__(root, transform, pre_transform)
        
        # 如果数据已经存在于磁盘上，直接加载
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        print("数据加载")
    @property
    def raw_file_names(self):
        # 无需使用原始文件，因此返回空列表
        return []

    @property
    def processed_file_names(self):
        # 定义处理后保存的文件名
        return ["data.pt"]

    def download(self):
        # 没有下载步骤，跳过
        pass
    
    def emb_node_edge(self,g):
        g.x = self.node_embs[g.x]
        g.edge_attr = self.edge_embs[g.xe]
        return g
    
    def process(self):
        """
        将传入的图数据列表进行处理，并保存到磁盘。
        """
        if self.data_list is not None:
            # 如果 pre_transform 被定义，应用到数据上
            
            self.data_list = [self.emb_node_edge(data) for data in self.data_list]
            
            # 将 data_list 打包成 PyTorch Geometric 格式的数据结构 (data, slices)
            data, slices = self.collate(self.data_list)
            # 保存处理后的数据
            torch.save((data, slices, ), self.processed_paths[0])

            


if __name__ == "__main__":
    root = f'/disk/NarGINA/dataset/ChildText/sum_score'
    llm_name = 'ST_ofa_768'
    llm_b_size = 128
    load_texts = False
    if 'vicuna' in llm_name or 'ofa' in llm_name:
        pooling = True
    else:
        pooling = False
    encoder = SentenceEncoder("ST", batch_size=llm_b_size,pooling=pooling)
    dataset = ChildTextDataset(load_texts=load_texts,root=root,llm_name=llm_name,encoder=encoder,add_prompt=False)
    dataset.print_split()
