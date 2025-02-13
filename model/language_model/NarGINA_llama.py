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


from typing import List, Optional, Tuple, Union
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch_geometric
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..NarGINA_arch import NarGINAMetaModel, NarGINAMetaForCausalLM
from utils.constants import IGNORE_INDEX


class NarGINAConfig(LlamaConfig):
    model_type = "llaga"


class NarGINALlamaModel(NarGINAMetaModel, LlamaModel):
    config_class = NarGINAConfig

    def __init__(self, config: LlamaConfig):
        super(NarGINALlamaModel, self).__init__(config)


class NarGINALlamaForCausalLM(LlamaForCausalLM, NarGINAMetaForCausalLM):
    config_class = NarGINAConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = NarGINALlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,# [bs,126]
        attention_mask: Optional[torch.Tensor] = None,# [bs,126]
        past_key_values: Optional[List[torch.FloatTensor]] = None,#none
        inputs_embeds: Optional[torch.FloatTensor] = None,#none
        labels: Optional[torch.LongTensor] = None,## [2,126]=[-100,-100,....,27308,29318,1983u,1..]
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,#2,111
        graph_emb: Optional[torch.FloatTensor] = None,#2,111,1135
        edge_index: Optional[torch.FloatTensor] = None,#2,111,1135
        edge_attr: Optional[torch.FloatTensor] = None,#2,111,1135
        edge_type: Optional[torch.FloatTensor] = None,#2,111,1135
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:# 执行完DataCollatorForSupervisedDataset，执行这个
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions#false
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )#false
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict#true

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb,edge_index=edge_index,edge_attr=edge_attr,edge_type=edge_type)
        #None，    [bs,236],        none,            [bs,236,4096],  [bs,236]
        
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]#[2,236,4096]
        logits = self.lm_head(hidden_states)#[2, 236, 32000]
        m = logits.min()

        loss = None
        if labels is not None:# WHY SHIFT? 在语言模型的训练中，给定一个序列，模型会通过当前词来预测下一个词。因此，模型在时间步 t 的输出（logits）应该与时间步 t+1 的标签（labels）对齐：
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
                "edge_index": kwargs.get("edge_index", None),
                "edge_attr": kwargs.get("edge_attr", None),
                "edge_type": kwargs.get("edge_type", None),                
            }
        )
        return model_inputs

AutoConfig.register("llaga", NarGINAConfig)
AutoModelForCausalLM.register(NarGINAConfig, NarGINALlamaForCausalLM)
