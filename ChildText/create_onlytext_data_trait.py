import json
import json
from prompt import LABEL_TEMPLATE_TRAIT, LABEL_TEMPLATE_TRAIT_COMMENT, PROMPT_TEMPLATE,PROMPT_TEMPLATE_TRAIT_COMMENT_FEW_SHOT
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
# 从 JSON 文件中读取数据
with open('/disk/NarGINA/ChildText/raw_graph/updated_total_data2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
with open('/disk/NarGINA/ChildText/raw_graph/split_index.json', 'r', encoding='utf-8') as file:
    split_indices = json.load(file)

train_data, validate_data, test_data =Subset(dataset=data,indices=split_indices['train']),Subset(dataset=data,indices=split_indices['valid']),Subset(dataset=data,indices=split_indices['test'])

def convert(data):
# 转换为指定的对话格式
    converted_data = []
    for index, item in enumerate(data):
        prompt = PROMPT_TEMPLATE_TRAIT_COMMENT_FEW_SHOT
        prompt = prompt.replace("<essay_text>",item['essay_text'])
        labels = LABEL_TEMPLATE_TRAIT.replace("<macro_score>",str(item['macro_score'])).replace("<micro_score>",str(item['micro_score'])).replace("<psych_score>",str(item['psych_score'])).replace("<total_score>",str(item['total_score']))
        identity = {
            "id": f"identity_{index}",
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": labels
                }
            ]
        }
        converted_data.append(identity)
    return converted_data

_train_data = convert(data=train_data)
_validate_data = convert(data=validate_data)
_test_data = convert(data=test_data)
# 保存数据集
with open('/disk/NarGINA/dataset/ChildText_onlytext/score_trait/few-shot/trait_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(_train_data, f, ensure_ascii=False, indent=4)

with open('/disk/NarGINA/dataset/ChildText_onlytext/score_trait/few-shot/trait_validate_data.json', 'w', encoding='utf-8') as f:
    json.dump(_validate_data, f, ensure_ascii=False, indent=4)

with open('/disk/NarGINA/dataset/ChildText_onlytext/score_trait/few-shot/trait_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(_test_data, f, ensure_ascii=False, indent=4)

print("数据已成功从读取，并保存。")
