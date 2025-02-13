import json
from collections import defaultdict

# 假设上传的文件路径是 graph_json_file
graph_json_file = '/disk/NarGINA/relation_extraction/data/graph.json'  # 请替换为实际文件路径

# 读取上传的 JSON 文件
with open(graph_json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 根据作文ID重新组织数据
grouped_data = defaultdict(lambda: {"events": [], "edges": [], "child_name": "", "text": ""})

# 遍历数据，按作文ID分组
for entry in data:
    # 获取事件和作文ID
    events = grouped_data[entry["作文ID"]]["events"]
    event1 = entry["事件1"]
    event2 = entry["事件2"]
    
    # 添加事件，去重
    if event1 not in events:
        events.append(event1)
    if event2 not in events:
        events.append(event2)

    # 组织边，只添加关系不为“无”的边
    if entry["关系"] != "无":
        edges = grouped_data[entry["作文ID"]]["edges"]
        head = events.index(event1)  # 获取事件1的索引
        tail = events.index(event2)  # 获取事件2的索引
        edges.append({
            "head": head,
            "tail": tail,
            "relation": entry["关系"]
        })

    # 保存作文ID和作文文本
    grouped_data[entry["作文ID"]]["child_name"] = entry["作文ID"]
    grouped_data[entry["作文ID"]]["text"] = entry["作文文本"]
# 将重新组织的数据转换成所需的格式
structured_data = []
for key, value in grouped_data.items():
    structured_data.append({
        "events": {str(i): event for i, event in enumerate(value["events"])},
        "edges": value["edges"],
        "child_name": value["child_name"],
        "text": value["text"]
    })

# 保存新的结构化数据为 JSON 文件
structured_file_path = '/disk/NarGINA/relation_extraction/data/processed_graph.json'  # 保存路径
with open(structured_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(structured_data, f_out, ensure_ascii=False, indent=4)

print(f"生成的文件已保存为: {structured_file_path}")
