import os
import json

def load_entity_descriptions(file_path):
    """加载 entity2text 文件，生成事件索引和描述的映射"""
    entity_descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            index, description = line.strip().split('\t', 1)
            entity_descriptions[index] = description
    return entity_descriptions

def parse_valid_file(valid_path, entity_descriptions):
    """解析 valid.txt 文件，生成事件关系描述"""
    relations = []
    with open(valid_path, 'r', encoding='utf-8') as file:
        for line in file:
            event1, relation, event2, essay_id = line.strip().split('\t')
            description1 = entity_descriptions.get(event1, f"{event1}")
            description2 = entity_descriptions.get(event2, f"{event2}")
            relations.append({
                '事件1': description1,
                '关系': relation,
                '事件2': description2,
                '作文ID': essay_id
            })
    return relations

def load_essays(json_path):
    """加载作文 JSON 文件，生成作文评分和文本映射"""
    essays = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            essays[entry['filename']] = {
                'filename': entry['filename'],
                'essay_text': entry['essay_text']
            }
    return essays

def find_matching_essay(essay_id, essays):
    """尝试通过作文ID部分匹配到对应的作文"""
    for key in essays:
        if essay_id in key:
            return essays[key]
    return None

def integrate_relations_with_essays(relations, essays):
    """整合事件关系和作文文本"""
    integrated_data = []
    for relation in relations:
        essay_info = essays.get(relation['作文ID'], None)
        if essay_info is None:
            print(f"未找到匹配的作文: 作文ID={relation['作文ID']}")
        integrated_data.append({
            '事件1': relation['事件1'],
            '关系': relation['关系'],
            '事件2': relation['事件2'],
            '作文ID': relation['作文ID'],
            '作文文本': essay_info.get('essay_text', '无文本') if essay_info else '无文本'
        })
    return integrated_data

def save_integrated_data(output_path, integrated_data):
    """保存整合后的数据到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(integrated_data, file, ensure_ascii=False, indent=4)

# 配置输入和输出文件路径
# 配置输入和输出文件路径
entity2text_path = '/disk/NarGINA/relation_extraction/data/entity2text.txt'  # 替换为实际路径
valid_path = '/disk/NarGINA/relation_extraction/data/xlw_test.txt'  # 替换为实际路径
json_path = '/disk/NarGINA/relation_extraction/data/text_data.json'  # 替换为实际路径
output_path = '/disk/NarGINA/relation_extraction/data/xlw_test_data_2.json'

# 加载数据
entity_descriptions = load_entity_descriptions(entity2text_path)
relations = parse_valid_file(valid_path, entity_descriptions)
essays = load_essays(json_path)

# 整合数据
integrated_data = integrate_relations_with_essays(relations, essays)

# 保存整合结果
save_integrated_data(output_path, integrated_data)
print(f"整合完成，结果已保存到 {output_path}")
