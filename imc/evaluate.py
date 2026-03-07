import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
'''
触发器评估
'''
def evaluate_trigger(full_model, trigger, images, target_label):
    """
    评估触发器效果：将触发器应用到干净图像上，测量模型对目标类别的平均响应
    """
    # 调整触发器大小以匹配图像
    trigger_resized = F.interpolate(
        trigger if trigger.dim() == 4 else trigger.unsqueeze(0),
        size=images.shape[2:],
        mode="bilinear",
        align_corners=False
    )
    
    # 应用触发器到所有图像
    if trigger_resized.shape[0] == 1 and images.shape[0] > 1:
        trigger_resized = trigger_resized.expand(images.shape[0], -1, -1, -1)
    alpha=0.2
    blended = (images + trigger_resized).clamp(0, 1)
    
    # 计算目标类别的平均概率
    with torch.no_grad():
        logits = full_model(blended)
        if logits.dim() > 2:
            logits = logits.mean(dim=[2, 3])
        probs = F.softmax(logits, dim=1)
        score = probs[:, target_label].mean().item()

    if score>=1:score-=1 # 特判


    return score