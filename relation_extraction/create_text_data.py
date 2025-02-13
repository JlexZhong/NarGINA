import json
import random

from sklearn.model_selection import train_test_split

# 从 JSON 文件中读取数据
with open('/disk/NarGINA/relation_extraction/data/train_data.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)
with open('/disk/NarGINA/relation_extraction/data/valid_data.json', 'r', encoding='utf-8') as file:
    val_data = json.load(file)
with open('/disk/NarGINA/relation_extraction/data/xlw_test_data_2.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

llm_prompt = """你的任务是对儿童讲述的故事《Frog, Where Are You?》（Mayer, 1969）中存在的两个事件，预测其事件关系。包括：并列,动机因果,心理因果,物理因果,使能因果和无关系。

事件1: <e_1>
事件2: <e_2>

注意：请你直接输出预测的事件关系，如果你认为两个事件之间没有关系，请输出“无”。。
Question：事件1和事件2之间的事件关系是？
"""

def convert(data, keep_no_ratio=0.02):
    # 转换为指定的对话格式
    converted_data = []
    no_relation_samples = []

    for index, item in enumerate(data):
        prompt = llm_prompt
        prompt = prompt.replace("<essay_text>", item['作文文本'])
        prompt = prompt.replace("<e_1>", item['事件1'])
        prompt = prompt.replace("<e_2>", item['事件2'])
        labels = str(item['关系'])
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
        if labels == '无':
            no_relation_samples.append(identity)
        else:
            converted_data.append(identity)

    # 随机抽取指定比例的 "无" 样本
    num_no_samples_to_keep = int(len(no_relation_samples) * keep_no_ratio)
    sampled_no_relation = random.sample(no_relation_samples, num_no_samples_to_keep)

    # 将抽样的 "无" 样本与其他样本合并
    converted_data.extend(sampled_no_relation)

    return converted_data

_train_data = convert(data=train_data, keep_no_ratio=0.02)
_validate_data = convert(data=val_data, keep_no_ratio=0.02)
_test_data = convert(data=test_data, keep_no_ratio=1)

# 保存数据集
with open('/disk/NarGINA/relation_extraction/dataset/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(_train_data, f, ensure_ascii=False, indent=4)

with open('/disk/NarGINA/relation_extraction/dataset/validate_data.json', 'w', encoding='utf-8') as f:
    json.dump(_validate_data, f, ensure_ascii=False, indent=4)

with open('/disk/NarGINA/relation_extraction/dataset/xlw_test_data_2.json', 'w', encoding='utf-8') as f:
    json.dump(_test_data, f, ensure_ascii=False, indent=4)

print("数据已成功读取并保存。")
