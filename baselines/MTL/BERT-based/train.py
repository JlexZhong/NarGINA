import os

from termcolor import colored

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import cohen_kappa_score
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
import torch.nn as nn
import os
from tqdm import tqdm

# ======================== 配置参数 ========================
BERT_PATH = "/disk/NarGINA/weights/bert-base-uncased"  # 你的 BERT 预训练模型路径
MODEL_SAVE_PATH = "/disk/NarGINA/baselines/MTL/An exploration of automated narrative analysis via machine learning/save"  # 你希望存储模型的文件夹
DATA_PATH = "/disk/NarGINA/ChildText/raw_graph/updated_total_data2.json"  # 你的 JSON 数据文件
SPLIT_PATH = "/disk/NarGINA/ChildText/raw_graph/split_index.json"  # 数据集划分 JSON
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-5

# ======================== 加载 JSON 数据 & 划分 ========================
def load_data(json_path, split_path):
    """从 JSON 文件加载数据，并按照 split_index.json 进行数据划分"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(split_path, 'r', encoding='utf-8') as file:
        split_indices = json.load(file)

    train_data = Subset(data, split_indices['train'])
    validate_data = Subset(data, split_indices['valid'])
    test_data = Subset(data, split_indices['test'])

    return train_data, validate_data, test_data

# 加载数据 & 按照 split_index.json 进行划分
train_data, val_data, test_data = load_data(DATA_PATH, SPLIT_PATH)

# ======================== 数据集类 ========================
class EssayDataset(Dataset):
    """数据集类，将文本转换为 BERT 输入格式"""

    def __init__(self, data, tokenizer, score_key):
        self.tokenizer = tokenizer
        self.texts = [entry["essay_text"] for entry in data]
        self.scores = [entry[score_key] for entry in data]  # 选定的评分维度

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        target = torch.tensor(self.scores[idx], dtype=torch.float)

        return input_ids, attention_mask, target

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ======================== 定义 BERT 回归模型 ========================
class BertForSequenceRegression(nn.Module):
    def __init__(self):
        super(BertForSequenceRegression, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, targets=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output).clamp(0, 10)  # 限制范围 0-10

        if targets is not None:
            loss = self.loss_fct(predictions.view(-1), targets.view(-1))
            return loss, predictions
        else:
            return predictions

# ======================== 训练 & 评估代码 ========================
def train(model, train_loader, val_loader, score_key):
    """训练 BERT 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids, attention_mask, targets = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, targets = [x.to(device) for x in batch]
                loss, _ = model(input_ids, attention_mask, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        save_path = os.path.join(MODEL_SAVE_PATH, f"best_model_{score_key}.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved for {score_key} at {save_path}!")

    print(f"Training complete for {score_key}!")

# ======================== 计算 QWK 指标 ========================
def evaluate_qwk(model, test_loader):
    """计算 Quadratic Weighted Kappa (QWK)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, targets = [x.to(device) for x in batch]
            preds = model(input_ids, attention_mask).view(-1)
            true_labels.extend(targets.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    qwk_score = cohen_kappa_score(np.round(true_labels), np.round(predictions), weights="quadratic")
    return qwk_score

# ======================== 训练 & 评估所有评分维度 ========================
score_keys = ["psych_score","macro_score", "micro_score", "total_score"]
qwk_scores = {}

for score_key in score_keys:
    print(f"Training model for {score_key}...")

    train_dataset = EssayDataset(train_data, tokenizer, score_key)
    val_dataset = EssayDataset(val_data, tokenizer, score_key)
    test_dataset = EssayDataset(test_data, tokenizer, score_key)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BertForSequenceRegression()
    train(model, train_loader, val_loader, score_key)

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f"best_model_{score_key}.pth")))
    qwk_scores[score_key] = evaluate_qwk(model, test_loader)
    print(f"{score_key} QWK: {qwk_scores[score_key]:.4f}")

average_qwk = np.mean(list(qwk_scores.values()))
print(f"Average QWK across all scores: {average_qwk:.4f}")
"""
Average QWK across all scores: 0.7216
"""