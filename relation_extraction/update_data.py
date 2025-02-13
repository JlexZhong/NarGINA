import os
import json

# 文件夹路径
folders = ['/disk/NarGINA/relation_extraction/data/event/train', '/disk/NarGINA/relation_extraction/data/event/eval', '/disk/NarGINA/relation_extraction/data/event/dev']

# 保存所有 JSON 字段的列表
all_data = []

# 遍历文件夹
for folder in folders:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):  # 确保只处理 JSON 文件
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    essay_text = ""
                    for line in f:
                        # 加载每行 JSON 数据
                        data = json.loads(line.strip())
                        # 拼接 sentence_text
                        essay_text += data["sentence_text"] + " "
                    
                    # 提取文件名
                    filename = os.path.splitext(os.path.basename(file))[0]
                    if filename.endswith(".txt.ann.event"):
                        filename = filename.replace(".txt.ann.event", "")
                    
                    # 添加到数据列表
                    all_data.append({
                        "filename": filename,
                        "essay_text": essay_text.strip()  # 去除多余的空格
                    })

# 保存为一个合并的 JSON 文件
output_file_path = '/disk/NarGINA/relation_extraction/data/text_data.json'
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    json.dump(all_data, out_file, ensure_ascii=False, indent=4)

print(f"All data has been merged and saved to {output_file_path}")
