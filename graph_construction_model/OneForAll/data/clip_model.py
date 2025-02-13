from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List
from pkg_resources import packaging

from OneForAll.data.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import Parameter
from torch import nn, optim
from tqdm.autonotebook import trange
from torch_scatter import scatter_mean
from sentence_transformers import SentenceTransformer

_tokenizer = _Tokenizer()


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class GNN(MessagePassing):
    def __init__(self, args, **kwargs):
        super(GNN, self).__init__(aggr='add', **kwargs)
        self.vars = nn.ParameterList()

        w = nn.Parameter(torch.ones([args.gnn_hid, args.gnn_input]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_hid)))

        w = nn.Parameter(torch.ones([args.gnn_output, args.gnn_hid]))
        torch.nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_output)))

    @staticmethod
    def norm(edge_index, num_nodes, improved=False, dtype=None):
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, vars=None):
        if vars is None:
            vars = self.vars
        improved = False

        w, b = vars[0], vars[1]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)

        w, b = vars[2], vars[3]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars


class CLIP(nn.Module):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef

        self.gnn = GNN(args)
        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size,
                                            args.transformer_width)  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.dtype = self.gnn.vars[0].dtype

        self.optim = optim.Adam([{'params': self.token_embedding.weight},
                                 {'params': self.positional_embedding},
                                 {'params': self.transformer.parameters()},
                                 {'params': self.text_projection},
                                 {'params': self.gnn.parameters()}
                                 ], lr=args.lr)
        self.batch_size = args.batch_size

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, idx_train, x, adj):
        embs = self.gnn(x, adj)
        train_embs = embs[idx_train]
        return train_embs
    
    def get_mentions(self,event):
        trigger = event.split('(')[0]
        arguments = event.split('(')[1].split(')')[0].split(';')

        return trigger,arguments

    def construct_node_adj_link_level(self,sentence,x,adj,index):
        event = sentence[0].split('description: ')[1]

        trigger,arguments = self.get_mentions(event)
        x.append(trigger)
        if len(index) == 0:
            index.append(0)
        else:
            index.append(index[len(index)-1]+1)
        index_trigger = len(x)-1
        for argument in arguments:
            x.append(argument)
            index.append(index[len(index)-1])
            adj.append([index_trigger,len(x)-1])
            adj.append([len(x)-1,index_trigger])

        return x,adj,index
    
    def construct_node_adj_node_level(self,sentence,x,adj,index):
        tmp = sentence[0].split('<first_event>')[1]
        first_event = tmp.split('<second_event>')[0]
        if len(tmp.split('<second_event>'))==1:
            print('!!!!')
        second_evend = tmp.split('<second_event>')[1]

        trigger_f,arguments_f = self.get_mentions(first_event)
        x.append(trigger_f)
        if len(index) == 0:
            index.append(0)
        else:
            index.append(index[len(index)-1]+1)
        index_trigger = len(x)-1
        for argument in arguments_f:
            x.append(argument)
            index.append(index[len(index)-1])
            adj.append([index_trigger,len(x)-1])
            adj.append([len(x)-1,index_trigger])

        trigger_s,arguments_s = self.get_mentions(second_evend)
        x.append(trigger_s)
        index.append(index[len(index)-1]+1)
        index_trigger = len(x)-1
        for argument in arguments_s:
            x.append(argument)
            index.append(index[len(index)-1])
            adj.append([index_trigger,len(x)-1])
            adj.append([len(x)-1,index_trigger])

        return x,adj,index
    

    def get_node_feature_chinese(self,node):

        model_path = 'OneForAll/cache_data/model/clip-model/ST_chinese_embedding/'
        model = SentenceTransformer(model_path)
        embeddings = model.encode(node)

        return torch.tensor(embeddings)

    def encode_graph(self, texts, to_tensor=True, level='link_level'):
        x = []
        adj = []
        index = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                # 构建节点和边,并指出那些节点属于这一句话
                if level == 'link_level':
                    self.construct_node_adj_link_level(sentences_batch,x,adj,index)
                else:
                    self.construct_node_adj_node_level(sentences_batch,x,adj,index)
                
            # 得到batch_node_feature
            node_feature = self.get_node_feature_chinese(x)
            # 传入gnn，得到子图编码
            embeddings = self.gnn(node_feature, torch.tensor(adj).transpose(0,1))
            embeddings = embeddings.cpu()
            if level == 'link_level':
                avg_embeddings = scatter_mean(embeddings, torch.tensor(index), dim=0)
                zero_padding = torch.zeros(avg_embeddings.size(0), 768)
                result = torch.cat((avg_embeddings, zero_padding), dim=1)
            else:
                avg_embeddings = scatter_mean(embeddings, torch.tensor(index), dim=0)
                assert avg_embeddings.size(0) % 2 == 0, "Number of rows in avg_result must be even"
                # 对每两个相邻的张量进行拼接
                concatenated_result = []
                for i in range(0, avg_embeddings.size(0), 2):
                    tmp = torch.cat([avg_embeddings[i].unsqueeze(0), avg_embeddings[i + 1].unsqueeze(0)], dim=1)
                    concatenated_result.append(tmp)
                concatenated_result = torch.cat(concatenated_result, dim=0)
                result = concatenated_result
                # concatenated_result = torch.cat([torch.cat((avg_embeddings[i], avg_embeddings[i + 1]), dim=1).unsqueeze(0) for i in range(0, avg_embeddings.size(0), 2)], dim=0)

        if not to_tensor:
            result = result.numpy()

        return result

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0,
                      2)  # NLD -> LND, batch_size * context_length *emb_dim -> context_length * batch_size  *emb_dim
        x = self.transformer(x)
        x = x.permute(1, 0,
                      2)  # LND -> NLD, context_length * batch_size *emb_dim -> batch_size * context_length *emb_dim
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot （end of token） embedding (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  #
        x = x @ self.text_projection
        return x

    def forward(self, x, adj, s_n, t_n, s_n_text, t_n_text, device, training=True):
        # 图节点特征
        s_image_features = self.encode_image(s_n, x, adj)
        # 节点文本特征
        s_text_features = self.encode_text(s_n_text)
        # 节点文本相邻节点的summary的特征
        t_text_features = self.encode_text(t_n_text)
        t_text_features = t_text_features.reshape(s_image_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)

        # normalized features
        s_image_features = s_image_features / s_image_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits

        labels = torch.arange(s_image_features.shape[0]).to(device)

        logit_scale = self.logit_scale.exp()  # the temporature hyperparameter
        logits = logit_scale * s_image_features @ s_text_features.t()
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        node_loss = (loss_i + loss_t) / 2

        logits = logit_scale * s_image_features @ t_text_features.t()
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        gt_loss = (loss_i + loss_t)/2

        logits = logit_scale * s_text_features @ t_text_features.t()
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        tt_loss = (loss_i + loss_t)/2

        all_loss = node_loss + self.edge_coef * gt_loss + self.edge_coef * tt_loss

        if training == True:
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            self.optim.step()

        # shape = [global_batch_size, global_batch_size]
        return round((all_loss.detach().clone()).cpu().item(), 4)


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:

    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

