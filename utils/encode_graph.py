import bitsandbytes as bnb
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from torch import nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from transformers import BitsAndBytesConfig
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel,AutoModelForCausalLM)
import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from torchmetrics import AveragePrecision, AUROC
from tqdm.autonotebook import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset



LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120,"bert-base-chinese":768,"llama2_7b_Chinese":4096,"llama2_7b_hf":4096,"vicuna-13b-v1.5":5120,"vicuna-7b-v1.5":4096}


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(self, llm_name, quantization=True, peft=True, cache_dir="/disk/NarGINA/weights", max_length=500):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name
        self.quantization = quantization

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        find all module for LoRA fine-tuning.
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, config)

        return model

    def get_llm_model(self):
        if self.llm_name == "llama2_7b_Chinese":
            model_name = "Llama2-Chinese-7b-Chat"
            # ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
            # ConfigClass = AutoConfig
            # TokenizerClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM

        elif self.llm_name == "vicuna-13b-v1.5":
            model_name = "vicuna-13b-v1.5"
            # ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
            # ConfigClass = AutoConfig
            # TokenizerClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM
        elif self.llm_name == "vicuna-7b-v1.5":
            model_name = "vicuna-7b-v1.5"
            # ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
            # ConfigClass = AutoConfig
            # TokenizerClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM
        elif self.llm_name == "llama2_7b_hf":
            model_name = "Llama-2-7b-hf"
            # ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
            # ConfigClass = AutoConfig
            # TokenizerClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "bert-base-chinese":
            model_name = "bert-base-chinese"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        model_path = self.cache_dir+'/'+model_name+'/'
        #model_path = "../cache_data/model/meta-llama/Llama-2-7b-hf"
        print(model_path)
        if self.quantization:
            bnb_config = self.create_bnb_config()
            # model = ModelClass.from_pretrained(model_name,
            #                                    quantization_config=bnb_config,
            #                                    #attn_implementation="flash_attention_2",
            #                                    #torch_type=torch.bfloat16,
            #                                    cache_dir=self.cache_dir)
            model = ModelClass.from_pretrained(model_path,
                                               quantization_config=bnb_config
                                               #attn_implementation="flash_attention_2",
                                               #torch_type=torch.bfloat16
                                               )
        else:
            print(self.llm_name)
            if "llama2" in self.llm_name or "vicuna" in self.llm_name:
                model = ModelClass.from_pretrained(model_path, device_map='balanced') 
            else:
                model = ModelClass.from_pretrained(model_path)
            

        tokenizer = TokenizerClass.from_pretrained(model_path, add_eos_token=True)

        if self.llm_name[:6] == "llama2" or self.llm_name[:6] == "vicuna":
            tokenizer.pad_token = tokenizer.bos_token

        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            #self.model.to('cuda')
            if "llama" and 'vicuna' not in self.llm_name:
                self.model.to('cuda')
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)
            
            return outputs, text_tokens["attention_mask"]


class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="/disk/NarGINA/weights", batch_size=1, multi_gpu=True,pooling = False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMModel(llm_name, quantization=False, peft=False, cache_dir=cache_dir)
        self.pooling = pooling
        # self.model.to(self.device)

    def encode(self, texts, promt_prefix=None,to_tensor=True):# node text list
        all_embeddings = []
        
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=256)
                text_tokens = text_tokens.to("cuda")
                embeddings, _ = self.model.encode(text_tokens, pooling=self.pooling)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()



def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        # device = torch.device("cuda:2")
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    # print("device:",device,gpu_ids)
    return device, gpu_ids


###########################                      DATASET                          ########################################


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

    def __init__(self, set_name: str = None, 
                 load_texts: bool = False, 
                 encoder = None,
                 root: str = "./cache_data", 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, 
                 llm_name =None,
                 add_prompt = None
                 ):

        self.set_name = set_name
        self.load_texts = load_texts
        self.root = root
        self.encoder = encoder
        self.llm_name = llm_name
        self.add_prompt = add_prompt
        if not self.load_texts:
            # assert self.encoder is not None
            suffix = llm_name
        else:
            suffix = 'raw'
        if self.set_name == None:
            self.data_dir = osp.join(self.root, suffix)
        else:
            self.data_dir = osp.join(self.root, str(self.set_name), suffix)

        super().__init__(self.data_dir, transform, pre_transform)
        safe_mkdir(self.data_dir)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.text = pth_safe_load(self.processed_paths[1])
        self.side_data = pth_safe_load(self.processed_paths[2])
        print("数据加载完毕")

    def data2vec(self, data: list[str]) -> torch.Tensor:
        r"""
        Encode a list of string to a len(data)-by-d matrix, where d is the output dimension of the LLM.
        """
        if self.encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        # if self.llm_name == "ST":
        #     embeddings = self.ST_emb_text(data).cpu().numpy()#TODO
        # else:
        embeddings = self.encoder.encode(data).cpu().numpy()
        return embeddings

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]


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

    def ST_emb_text(self,texts):
        encoder = SentenceTransformer('/disk/NarGINA/weights/sentence-transformers')
        res = []
        for t in texts:
            text_emb = encoder.encode(t,convert_to_numpy=True,show_progress_bar=True)
            res.append(text_emb)

        return res

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(t) for t in texts]
    
    def text2feature_prompt_prefix(self,texts):
        [u_node_texts_lst, u_edge_texts_lst, prompt_edge_text, prompt_text] = texts
        prompt_node = '事件可表述为：谓语(主语;宾语;时间状语;地点状语)。这是一个事件：'
        prompt_edge = '这是一对事件之间的关系：'
        node_texts = [prompt_node + item for item in u_node_texts_lst]
        edge_texts = [prompt_edge + item for item in u_edge_texts_lst]
        node_embs = self.data2vec(node_texts)
        edge_embs = self.data2vec(edge_texts)
        return [node_embs,edge_embs]



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
            if self.llm_name == "ST_no_ofa":
                texts_emb = self.ST_emb_text(texts)
            else:
                if self.add_prompt:
                    texts_emb = self.text2feature_prompt_prefix(texts)
                else:
                    texts_emb = self.text2feature(texts)
            data, slices = self.add_text_emb(data_list, texts_emb)

        print("Saving...")
        torch.save((data, slices, ), self.processed_paths[0], pickle_protocol=4)

    def print_split(self):
            # 验证每个子集中的标签分布是否一致
        labels = [self.get(i).y.item() for i in range(len(self))]
        
        train_labels = [labels[i] for i in self.side_data['train']]
        valid_labels = [labels[i] for i in self.side_data['valid']]
        test_labels = [labels[i] for i in self.side_data['test']]

        print("训练集标签分布:", np.bincount(train_labels))
        print("验证集标签分布:", np.bincount(valid_labels))
        print("测试集标签分布:", np.bincount(test_labels))


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
        the index to prompt_edge_text_feat.#TODO

        """
        pass

    def get_prompt_text_feat(self, task_name):
        """
        Return the list of prompt node/edge feature for the task.
        """
        # TODO e2e graph
        task_map = self.get_task_map() # {e2egraph:{},lr_graph:{}}
        if task_name not in task_map:
            raise NotImplementedError(
                "Task " + task_name + " is not implemented for " + self.set_name + " dataset the implemented tasks are "
                + str(
                    task_map.keys()))
        feat_ind = task_map[task_name] # {noi{},edge{},classnode{}}
        prompt_feats = {}
        for k in feat_ind:#self.data中有eaasy——text
            a = getattr(self.data, feat_ind[k][0]) # [2,4096]
            b = a[feat_ind[k][1]] # [1,4096]
            if b.shape[0] == 768 or b.shape[0] == 4096: # TODO
                b = np.expand_dims(b,axis=0)
            prompt_feats[k] = b
        return prompt_feats # 三个都是1,4096



