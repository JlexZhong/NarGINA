#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn
import re

# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_vision_projector


from model.my_model import GraphAttModel, PyGRGCNEdge
from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
def check_lora_weight(adapter_path):
    state_dict = torch.load(adapter_path)

    # 创建一个新的字典，只修改键的名字
    new_state_dict = {key.replace("module.", "", 1) if key.startswith("module.") else key: value 
                    for key, value in state_dict.items()}
    return new_state_dict
    # # 保存新的权重文件
    # torch.save(new_state_dict, adapter_path)

    # #print("键名修改完成，新的权重文件已保存到")

def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))
    print("hidden_dim",hidden_dim)
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_dim)
    if projector_type == 'attn_mlp':
        print("config.mm_hidden_size",config.mm_hidden_size)
        return nn.Sequential(
                nn.Linear(config.mm_hidden_size, hidden_dim),  # 输入层到第一个隐藏层
                nn.GELU(),                               
                nn.Linear(hidden_dim, hidden_dim),     # 第一个隐藏层到第二个隐藏层
                nn.GELU(),                              
                nn.Linear(hidden_dim, hidden_dim),     # 第二个隐藏层到第三个隐藏层
            )
    if projector_type == 'attn_mlp_v2':
        print("config.mm_hidden_size",config.mm_hidden_size)
        return nn.Sequential(
                nn.Linear(config.mm_hidden_size, hidden_dim // 4),  # 输入层到第一个隐藏层
                nn.GELU(),                               
                nn.Linear(hidden_dim // 4, hidden_dim // 2),     # 第一个隐藏层到第二个隐藏层
                nn.GELU(),                              
                nn.Linear(hidden_dim // 2, hidden_dim),     # 第二个隐藏层到第三个隐藏层
            )
    
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')

def build_graph_encoder(config):
    llama_hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))
    model = GraphAttModel(
        llm_name=os.path.basename(config.name_or_path),
        outdim=llama_hidden_dim // 2 ,
        add_rwpe=None,
        num_heads=config.graph_encoder_num_heads,
        gnn_in_dim=config.graph_encoder_num_hidden,
        gnn_out_dim=config.graph_encoder_num_hidden,
        gnn_num_layers=config.graph_encoder_num_layers
    )
    config.mm_hidden_size = llama_hidden_dim // 2
    return model

class NarGINAMetaModel:

    def __init__(self, config):
        super(NarGINAMetaModel, self).__init__(config)
        if hasattr(config, "graph_encoder_type") and getattr(config, "graph_encoder_type") is not None and getattr(config, "graph_encoder_type") != "None":
            self.graph_encoder = build_graph_encoder(config)
        if hasattr(config, "mm_hidden_size"):
            self.mm_projector = build_graph_projector(config)
        if hasattr(config, "mm_use_graph_special_token") and getattr(config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()


    def initialize_graph_modules(self, model_args, fsdp=None):
        pretrain_mm_mlp_adapter = getattr(model_args, 'pretrain_mm_mlp_adapter', None)
        pretrain_graph_encoder_weights  = getattr(model_args, 'pretrain_graph_encoder_weights', None)
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(model_args, 'mm_hidden_size')
        # if self.config.mm_projector_type == 'attn_linear':
        #     self.config.head_num = getattr(model_args, 'head_num',)
        ## TODO 加入graph encoder
        if hasattr(model_args, "graph_encoder_type") and getattr(model_args, "graph_encoder_type") is not None and getattr(model_args, "graph_encoder_type") != "None":
            self.graph_encoder = build_graph_encoder(self.config)
        self.mm_projector = build_graph_projector(self.config)
        ## 
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            #false
            self.special_token_emb = self.build_special_tokens()

        # if pretrain_mm_mlp_adapter is not None:
        #     mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        #     def get_w(weights, keyword):
        #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        #     self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        if model_args.pretrain_mm_mlp_adapter is not None:
            # model.get_model().initialize_graph_modules(cfg_pretrained)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            if os.path.exists(model_args.pretrain_mm_mlp_adapter):
                mm_projector_weights = check_lora_weight(model_args.pretrain_mm_mlp_adapter)
                #mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                #mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                #print(mm_projector_weights)
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)
                print(f"Load MLP Projector pretrained weights from {model_args.pretrain_mm_mlp_adapter}")
        if  pretrain_graph_encoder_weights is not None:
            graph_encoder_weights = torch.load(pretrain_graph_encoder_weights, map_location='cpu')
            self.graph_encoder.load_state_dict(graph_encoder_weights,strict=False)
            print("Load graph encoder weights from ",pretrain_graph_encoder_weights)
            
    def build_special_tokens(self):
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            num_token=self.config.use_hop+2
            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).unsqueeze(1).detach()
            special_token_emb=torch.nn.parameter.Parameter(data=input_embeddings_avg.repeat(num_token, 1, 1), requires_grad=True)
            return special_token_emb
        return None

class NarGINAMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def insert_features(self, graph_index, x_list, hidden_size):
        """
        Parameters:
        - graph_index: Tensor of shape [graph_num, max_len], containing 100 or PAD.
        - x_list: List of feature matrices for each graph, each of shape [node_num, hidden_size].
        - hidden_size: The size of the hidden features for each node.

        Returns:
        - result: Tensor of shape [graph_num, max_len, hidden_size] where the features are inserted.
        """
        graph_num, max_len = graph_index.shape
        # 初始化结果矩阵，全0，大小为 [graph_num, max_len, hidden_size]
        dtype = x_list[0].dtype
        result = torch.zeros(graph_num, max_len, hidden_size,dtype=dtype).cuda()

        for i in range(graph_num):
            x = x_list[i]  # 当前图的节点特征矩阵
            node_idx = 0  # 用于跟踪节点特征在 x 中的索引
            for j in range(max_len):
                if graph_index[i, j] != DEFAULT_GRAPH_PAD_ID:  # 插入节点特征
                    result[i, j] = x[node_idx]  # 插入对应的节点特征
                    node_idx += 1
                # 否则就是 PAD 情况，默认 result[i, j] 为 0（已经初始化为全零）
        return result


    def encode_graphs(self, graph, graph_emb,edge_index=None,edge_attr=None,edge_type=None):
        #【bs,111],[bs,111,hidden]

        if hasattr(self.get_model(),'graph_encoder'):
            # FIXME注意,这里需要把填充的0或者-1作处理，在传入encoder
            #TODO
            graph_emb = self.get_model().graph_encoder(x=graph_emb,edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type)
            #[node-num,emd]-->[bs,111,hidden]
            num_graphs = graph_emb.size(0)
            assert num_graphs == graph.size(0)
            x_list = []
            for graph_idx in range(num_graphs):
                x_list.append(graph_emb[g.batch == graph_idx]) # FIXME
            graph_emb = self.insert_features(graph_index=graph,x_list=x_list,hidden_size=graph_emb.size(1))


        graph_features = self.get_model().mm_projector(graph_emb)# 【2,111,4096】
        graph_features[graph==DEFAULT_GRAPH_PAD_ID] = 0. # 把pad位置设为0
        return graph_features

    def inject_special_token(self, graph_emb):
        use_hop=self.config.use_hop
        sample_size = self.config.sample_neighbor_size
        assert graph_emb.shape[-2] == int((sample_size ** (use_hop + 1) - 1) / (sample_size - 1))
        assert self.model.special_token_emb.shape[0] == use_hop + 2
        new_graph_emb = []
        new_graph_emb.append(self.model.special_token_emb[0])
        cur=0
        for i in range(use_hop+1):
            cur_size = sample_size**i
            new_graph_emb.append(graph_emb[cur:cur+cur_size])
            cur+=cur_size
            new_graph_emb.append(self.model.special_token_emb[i+1])
        new_graph_emb = torch.concat(new_graph_emb, dim=0)
        return new_graph_emb

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb,edge_index=None,edge_attr=None,edge_type=None
    ):#past_key_values=none
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        # if getattr(self.config, 'mm_projector_type', 'linear') == 'attn_linear':
        #     graph_emb = graph_emb.view(graph_emb.size(0), graph_emb.size(1), self.config.mm_hidden_size * self.config.head_num)  # 形状变为 [batchsize, node-num, head-num * head-dim]

        graph_features = self.encode_graphs(graphs, graph_emb,edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type)#TODO 使用mm——projector转为graphtoken，[batchsize, 111, 4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]#shape=【1】,tensor=[43] 指示原文本中<graph>的位置
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:#numel return elements total num
                cur_graph_features = graph_features[cur_graph_idx]
                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)

                graph_token_start = graph_token_indices[0]# 43
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start+1])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:# 这里把graph token组合进input token
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))# 【43,4096】编码前半部分token
                    cur_new_input_embeds.append(cur_graph_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))# 【111】全是-100
                        cur_labels = cur_labels[graph_token_start+1:]
                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:# 这里
            new_input_embeds = torch.stack(new_input_embeds, dim=0)# [2, 236, 4096]
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)# 【2,236】

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def prepare_inputs_labels_for_multimodal_with_pad_mask(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb
    ):
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        graph_features = self.encode_graphs(graphs, graph_emb)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_attention_masks = []
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_attention_mask = attention_mask[batch_idx]
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_graph = graphs[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_attn_masks=[]
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                cur_graph = graphs[cur_graph_idx]
                cur_graph_mask = (cur_graph != DEFAULT_GRAPH_PAD_ID)
                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)

                graph_token_start = graph_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2]))
                    cur_attn_masks.append(cur_attention_mask[:graph_token_start])
                    cur_attn_masks.append(cur_graph_mask)
                    cur_attn_masks.append(cur_attention_mask[graph_token_start+1:graph_token_start+2])
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start+1])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_attn_masks.append(cur_attention_mask[:graph_token_start])
                    cur_attn_masks.append(cur_graph_mask)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[graph_token_start+1:]

                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                    cur_attention_mask = cur_attention_mask[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                    cur_attention_mask = cur_attention_mask[graph_token_start + 1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                cur_attn_masks.append(cur_attention_mask)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_attn_masks = [x.to(device=self.device) for x in cur_attn_masks]
            cur_attn_masks = torch.cat(cur_attn_masks, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_attention_masks.append(cur_attn_masks)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(new_attention_masks, _new_labels, new_labels):
                    assert cur_attention_mask.shape == cur_new_labels.shape
                    # new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            attention_mask = torch.stack(new_attention_masks, dim=0)
            assert attention_mask.shape == new_input_embeds.shape[:2]
            # if attention_mask is not None:
            #     new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            #     attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            #     assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_graph_tokenizer(self, model_args, tokenizer):

        if model_args.mm_use_graph_start_end:#false
            num_new_tokens = tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                #加载预训练mlp
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
