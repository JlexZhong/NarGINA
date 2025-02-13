import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
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


# 从匹配数据中提取节点文本 X
def extract_node_text(matched_data):
    events = matched_data.get("events", {})
    X = [events[key] for key in sorted(events.keys(), key=int)]  # 按照键的顺序提取文本
    return X

def extract_edge_data(matched_data):
    events = matched_data.get("events", {})
    edges = matched_data.get("edges", [])
    
    # 获取重新索引后的节点列表及其索引映射
    node_keys = sorted(events.keys(), key=int)  # 重新排序事件的 key
    node_index_map = {int(key): idx for idx, key in enumerate(node_keys)}  # 映射原始 key 到新索引

    edge_index = []
    edge_feature = []

    for edge in edges:
        if "head" in edge and "tail" in edge:
            original_head = edge["head"]
            original_tail = edge["tail"]

            # 将原始 head 和 tail 映射到新的节点索引
            if original_head in node_index_map and original_tail in node_index_map:
                new_head = node_index_map[original_head]
                new_tail = node_index_map[original_tail]

                relation_type = edge.get("relation", "")
                if EDGE_TYPE_MAPPING.get(relation_type, -1) != -1:  # 仅保留合法的关系类型
                    edge_index.append([new_head, new_tail])
                    edge_feature.append(relation_type)

    # 转为 PyTorch 张量
    edge_index = torch.from_numpy(np.array(edge_index).T).long() if edge_index else torch.empty((2, 0), dtype=torch.long)
    return edge_index, edge_feature


def load_xlw_text_dataset(raw_data_path,test_graph_path):
    # Read data into huge `Data` list.
    graphs = []
    with open(test_graph_path, 'r', encoding='utf-8') as file:
            raw_graph_data = json.load(file)
    with open(os.path.join(raw_data_path,'split_index.json'), 'r', encoding='utf-8') as file:
        split_index = json.load(file)               
    with open(os.path.join(raw_data_path,'updated_total_data2.json'), 'r', encoding='utf-8') as file:
        reader = json.load(file)   
        for id , row in enumerate(reader): 
            if id not in split_index['test']:
                continue
            filename = row['filename']
            matched_data = [item for item in raw_graph_data if item.get('child_name') == filename or item.get('child_name') in filename or filename in item.get('child_name')]
            
            if matched_data == []:
                print(filename,"无对应数据!!!!!!!!!!!!")
                continue
            matched_data = matched_data[0]
            source_text = row['essay_text']
            comment = row['comment'] if row['comment'] is not None else -1
            macro_score = row['macro_score'] if row['macro_score'] is not None else -1
            micro_score = row['micro_score'] if row['micro_score'] is not None else -1
            psych_score = row['psych_score'] if row['psych_score'] is not None else -1
            total_score = row['total_score']

            node_text = extract_node_text(matched_data)
            edge_index, edge_text = extract_edge_data(matched_data)

            # if edge_index.size(0) == 0 or edge_index.size(1) <= 1 or edge_index.max() >=len(node_text):
            #     print("错误数据")
            #     continue

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
            graph['split'] = 'test'

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


class ChildTextDataset_for_test(OFAPygDataset):
    
    def gen_data(self):
        raw_data_path = '/disk/NarGINA/ChildText/raw_graph'
        test_graph_path = "/disk/NarGINA/ChildText/raw_graph/processed_graph_t5uie_vicuna.json"
        pyg_graph, texts, split = load_xlw_text_dataset(raw_data_path,test_graph_path)
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



if __name__ == "__main__":
    root = f'/disk/NarGINA/dataset/ChildText_test/teacher_2/t5uie-vicuna_graph'
    llm_name = 'ST_ofa_768'
    llm_b_size = 128
    load_texts = False
    if 'vicuna' in llm_name or 'ofa' in llm_name:
        pooling = True
    else:
        pooling = False
    encoder = SentenceEncoder("ST", batch_size=llm_b_size,pooling=pooling)
    dataset = ChildTextDataset_for_test(load_texts=load_texts,root=root,llm_name=llm_name,encoder=encoder,add_prompt=False)
    dataset.print_split()
