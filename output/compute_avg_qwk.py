import os

# 文件夹路径（替换为你的实际路径）
folder_path = '/disk/NarGINA/output/ASAP++/only_text/answer_ASAP++_vicuna7b_onlytext_score_trait_lora'
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        # 提取文件名中的QWK值
        parts = filename.split('-')
        qwk_values = []
        for part in parts:
            if part.startswith("QWK_") and ".json" in part:
                try:
                    qwk_values.append(float(part.split('=')[1].replace('.json', '')))
                except (IndexError, ValueError):
                    continue
            elif part.startswith("QWK_"):
                try:
                    qwk_values.append(float(part.split('=')[1]))
                except (IndexError, ValueError):
                    continue
        # 计算平均QWK
        if qwk_values:
            average_qwk = sum(qwk_values) / len(qwk_values)

            # 构造新文件名
            avg_qwk_str = f"{average_qwk:.3f}"
            new_filename = f"{filename.rstrip('.json')}-QWK_avg={avg_qwk_str}.json"

            # 重命名文件
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
print("文件重命名完成！")
