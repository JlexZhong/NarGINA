import json

# 假设文件1和文件2的路径分别是 file_path_1 和 file_path_2
file_path_1 = '/disk/NarGINA/output/relation_preds/vicuna_test.json'  # 请替换为实际文件路径
file_path_2 = '/disk/NarGINA/relation_extraction/data/xlw_test_data_2.json'  # 请替换为实际文件路径

# 读取文件1
with open(file_path_1, 'r', encoding='utf-8') as f1:
    file1_data = [json.loads(line.strip()) for line in f1.readlines()]

# 读取文件2
with open(file_path_2, 'r', encoding='utf-8') as f2:
    file2_data = json.load(f2)

# 将文件1中的 "prediction" 赋值到文件2中的 "关系"
for i in range(len(file1_data)):
    file2_data[i]["关系"] = file1_data[i]["prediction"]

# 保存修改后的文件2
updated_file_path = '/disk/NarGINA/output/relation_preds/graph.json'  # 更新后的文件保存路径
with open(updated_file_path, 'w', encoding='utf-8') as f2_out:
    json.dump(file2_data, f2_out, ensure_ascii=False, indent=4)

print(f'更新后的文件已保存至 {updated_file_path}')
