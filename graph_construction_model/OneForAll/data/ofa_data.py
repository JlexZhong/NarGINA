import os
import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Callable, Any

import argparse

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset

from OneForAll.data.clip_model import CLIP

from OneForAll.utils import SentenceEncoder


def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)


def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path)
    return None


class OFAPygDataset(InMemoryDataset, ABC):
    r"""
    Base dataset class for OFA datasets. OFAPygDataset takes care of
    1, dataset loading
    2, text to feature transformation using LLM if specified.
    Currently, the class support two modes controlled by load_text. If load_text is true, the class
    only load raw texts into model. Otherwise, an LLM encoder must be specified to transform raw texts
    to feature.
    """

    def __init__(self, name: str, load_texts: bool, encoder: Optional[SentenceEncoder] = None,
                 root: str = "./cache_data", transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, if_subgraph = True,level='link_level'):

        self.name = name
        self.load_texts = load_texts
        self.root = root
        self.encoder = encoder

        self.if_subgraph = if_subgraph
        self.level = level

        # subgraph的encoder

        self.subgraph_encoder = CLIP(self.get_subgraphModel_args())
        self.subgraph_encoder.load_state_dict(torch.load('OneForAll/cache_data/model/clip-model/node_ttgt_8&12_0.1.pkl'.format('my_data_4')))

        if not self.load_texts:
            assert self.encoder is not None
            suffix = self.encoder.llm_name
        else:
            suffix = 'raw'
        self.data_dir = osp.join(self.root, self.name, suffix)

        super().__init__(self.data_dir, transform, pre_transform)
        safe_mkdir(self.data_dir)

        # # load text to the dataset instance
        # if self.load_texts:
        #     self.texts = torch.load(self.processed_paths[1])

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.side_data = pth_safe_load(self.processed_paths[2])

    def get_subgraphModel_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
        parser.add_argument('--ft_epoch', type=int, default=128, help='fine-tune epoch')
        parser.add_argument('--lr', type=float, default=2e-5)

        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--gnn_input', type=int, default=768)
        parser.add_argument('--gnn_hid', type=int, default=1024)
        parser.add_argument('--gnn_output', type=int, default=768)

        parser.add_argument('--edge_coef', type=float, default=0.1)
        parser.add_argument('--neigh_num', type=int, default=3)

        parser.add_argument('--num_labels', type=int, default=6)
        parser.add_argument('--k_spt', type=int, default=5)
        parser.add_argument('--k_val', type=int, default=5)
        parser.add_argument('--k_qry', type=int, default=50)
        # parser.add_argument('--n_way', type=int, default=5)
        parser.add_argument('--n_way', type=int, default=6)

        parser.add_argument('--context_length', type=int, default=128)
        parser.add_argument('--coop_n_ctx', type=int, default=4)
        parser.add_argument('--prompt_lr', type=float, default=0.01)

        parser.add_argument('--position', type=str, default='end')
        parser.add_argument('--class_specific', type=bool, default=False)
        parser.add_argument('--ctx_init', type=bool, default=True)

        parser.add_argument('--embed_dim', type=int, default=768)
        parser.add_argument('--transformer_heads', type=int, default=8)
        parser.add_argument('--transformer_layers', type=int, default=12)
        parser.add_argument('--transformer_width', type=int, default=512)
        parser.add_argument('--vocab_size', type=int, default=49408)
        parser.add_argument('--gpu', type=int, default=1)

        args = parser.parse_args([])

        return args

    def data2vec(self, data: list[str]) -> torch.Tensor:
        r"""
        Encode a list of string to a len(data)-by-d matrix, where d is the output dimension of the LLM.
        """
        if self.encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        
        embeddings_ofa_lm = self.encoder.encode(data).cpu().numpy()
        # embeddings_ofa_lm = self.encoder.encode(data).cuda().numpy()
        
        return embeddings_ofa_lm

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(t) for t in texts]

    @abstractmethod
    def gen_data(self) -> tuple[list[pyg.data.Data], list[list[str]], Any]:
        r"""
        Subclass should implement this method, it should generate
        1, a list of pyg.data.Data graphs
        2, a list of str sets that can be processed by self.text2feature
        3, any side data that should be stored during preprocessing
        The list of string (2) will be processed by encoder to vector representation and be fed into
        self.add_text_emb along with the list of graphs.
        """
        pass


    @abstractmethod
    def add_raw_texts(self, data_list, texts: list[str]) -> tuple[pyg.data.Data, Mapping]:
        r"""
        Args:
            data_list: a list of pyg.data.Data generated by self.gen_data
            texts: a list of text generated by self.gen_data

        Returns:

        """
        pass

    @abstractmethod
    def add_text_emb(self, data_list, texts_emb: list[torch.Tensor]) -> tuple[pyg.data.Data, Mapping]:
        r"""
        Args:
            data_list: a list of pyg.data.Data generated by self.gen_data
            texts_emb: a list of torch.Tensor generated by self.encoder from the text generated by self.gen_data

        Returns:

        """
        pass


    def process(self):

        data_list, texts, side_data = self.gen_data()

        torch.save(texts, self.processed_paths[1])
        if side_data is not None:
            torch.save(side_data, self.processed_paths[2])
        else:
            torch.save("No side data", self.processed_paths[2])

        if self.load_texts:
            data, slices = self.add_raw_texts(data_list, texts)
        else:
            if self.encoder.model is None:
                self.encoder.get_model()
            
            if self.if_subgraph:
                # 子图嵌入            
                subgraph_emb = self.subgraph_encoder.encode_graph(texts[0],level=self.level).cpu().numpy()
                # subgraph_emb = self.subgraph_encoder.encode_graph(texts[0]).cuda().numpy()
                texts_emb = self.text2feature(texts)

                combined_emb = [torch.cat([torch.tensor(texts_emb[0]),torch.tensor(subgraph_emb)],dim=1).numpy()]

                for index in range(1,len(texts_emb)):
                    tmp1 = torch.tensor(texts_emb[index])
                    tmp2 = torch.tensor(subgraph_emb)
                    tmp3 = torch.zeros(tmp1.shape[0],tmp2.shape[1])

                    combined_emb.append(torch.cat([tmp1,tmp3],dim=1).numpy())

                data, slices = self.add_text_emb(data_list, combined_emb)
            
            else:

                # 不做子图嵌入            
                # subgraph_emb = self.subgraph_encoder.encode_graph(texts[0]).cpu().numpy()
                # subgraph_emb = self.subgraph_encoder.encode_graph(texts[0]).cuda().numpy()
                texts_emb = self.text2feature(texts)

                combined_emb = [torch.cat([torch.tensor(texts_emb[0]),torch.zeros(len(texts[0]),768)],dim=1).numpy()]

                for index in range(1,len(texts_emb)):
                    tmp1 = torch.tensor(texts_emb[index])
                    tmp2 = torch.tensor(subgraph_emb)
                    tmp3 = torch.zeros(tmp1.shape[0],tmp2.shape[1])

                    combined_emb.append(torch.cat([tmp1,tmp3],dim=1).numpy())

                data, slices = self.add_text_emb(data_list, combined_emb)

        print("Saving...")
        torch.save((data, slices, ), self.processed_paths[0], pickle_protocol=4)

    @abstractmethod
    def get_task_map(self) -> dict[str, dict]:
        """
        :return: a task map specifying the text feature used by different tasks.
        """
        pass

    @abstractmethod
    def get_edge_list(self, mode="e2e") -> dict[str, list]:
        """
        Return the edge construction protocol for different tasks.
        Args:
            mode: a string representing the task

        Returns: a dictionary whose keys are the connection types including
            "f2n": feature to noi node
            "n2f": noi node to feature
            "n2c": noi node to class node
            "c2n": class node to noi node
        The values are lists of length 2. first element is the edge type, second element is
        the index to prompt_edge_text_feat.

        """
        pass

    def get_prompt_text_feat(self, task_name):
        """
        Return the list of prompt node/edge feature for the task.
        """
        task_map = self.get_task_map()
        if task_name not in task_map:
            raise NotImplementedError(
                "Task " + task_name + " is not implemented for " + self.name + " dataset the implemented tasks are "
                + str(
                    task_map.keys()))
        feat_ind = task_map[task_name]
        prompt_feats = {}
        for k in feat_ind:
            prompt_feats[k] = getattr(self.data, feat_ind[k][0])[feat_ind[k][1]]
        return prompt_feats


class OFAPygSTDataset(OFAPygDataset):
    def __init__(self, name, encoder, root="./cache_data", load_text=False, load_feat=True, transform=None,
            pre_transform=None, meta_dict=None, ):

        self.name = name
        self.encoder = encoder
        self.root = root
        self.data_dir = osp.join(self.root, self.name)
        safe_mkdir(self.data_dir)
        super().__init__(self.data_dir, transform, pre_transform)

        if load_text:
            self.texts = torch.load(self.processed_paths[0])

        self.side_data = pth_safe_load(self.processed_paths[1])
        self.global_data = pth_safe_load(self.processed_paths[2])

        self.convert_data()

    @property
    def processed_file_names(self):
        return ["texts.pkl", "data.pt", "global_data.pt", "node_feat.npy", "edge_feat.npy", ]

    def len(self):
        return 0

    def convert_data(self):
        pass

    def process(self):
        if self.encoder.model is None:
            self.encoder.get_model()
        data_list, texts, side_data = self.gen_data()
        texts_emb = self.text2feature(texts)
        torch.save(texts, self.processed_paths[0])
        if side_data is not None:
            torch.save(side_data, self.processed_paths[1])
        else:
            torch.save("No side data", self.processed_paths[1])
        data, global_data = self.add_text_emb(data_list, texts_emb)
        if global_data is not None:
            torch.save(global_data, self.processed_paths[2])
        else:
            torch.save("No global data", self.processed_paths[2])

        print("Saving...")

        node_memmap = np.memmap(self.processed_paths[3], dtype="float32", mode="w+", shape=data[0].shape, )
        node_memmap[:] = data[0]
        node_memmap.flush()

        edge_memmap = np.memmap(self.processed_paths[4], dtype="float32", mode="w+", shape=data[1].shape, )
        edge_memmap[:] = data[1]
        edge_memmap.flush()

    def get(self, idx):
        data = torch.load(self.processed_paths[idx + 3])
        return data
