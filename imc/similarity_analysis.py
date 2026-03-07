import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

'''
类别相似度分析
'''
def find_far(classifier):
    """
    贪心算法：为每个类别找到最不相似的类别
    返回: [(目标类别, 源类别, 相似度), ...]
    """
    # 获取分类器权重矩阵 [num_classes, feature_dim]
    weight = classifier.weight.data
    print(f"分类器权重形状: {weight.shape}")
    
    # 计算余弦相似度矩阵
    norm_w = weight / weight.norm(dim=1, keepdim=True)
    sim_matrix = norm_w @ norm_w.t()  # [num_classes, num_classes]
    
    # 掩码对角线（排除自身）
    num_classes = weight.shape[0]
    diag_mask = torch.eye(num_classes, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix.masked_fill_(diag_mask, float('inf'))
    
    # 找到每个类别最不相似的类别
    results = []
    for i in range(num_classes):
        j = torch.argmin(sim_matrix[i]).item()  # 最小相似度的索引
        sim_val = sim_matrix[i, j].item()       # 最小相似度值
        results.append((i, j, sim_val))
    
    return results