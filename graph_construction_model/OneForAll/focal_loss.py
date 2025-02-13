import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=2,reduction='mean'):
        """
        :param alpha: 计算损失时的平衡因子
        :param gamma: 聚焦因子
        :param reduction: 指定返回的损失值的计算方式, 可以是 'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的输出,大小为 (N,C) 的张量,其中 N 是批量大小,C 是类别数
        :param targets: 目标标签,大小为 (N,) 的张量,包含对应的类别索引
        """
        # 使用 softmax 将模型输出转化为概率
        inputs = inputs.view(-1,self.num_classes)
        p = nn.functional.softmax(inputs, dim=-1)
        
        # 获取目标类别的概率
        # targets = targets.view(-1, self.num_classes)
        # targets = targets.argmax(dim=-1).view(-1,1)
        targets = targets.view(-1,1)
        p_t = p.gather(1, targets).view(-1)
        
        # 计算 Focal Loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * p_t.log()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss