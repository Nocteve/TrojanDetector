import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
'''
模型加载和配置
'''
def load_classification_model(model_dir):
    """加载分类模型和配置"""
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.pt")
    
    # 1. 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    architecture_name = config['MODEL_ARCHITECTURE']
    img_size = config['IMG_SIZE']
    num_classes = config['NUMBER_CLASSES']
    
    print(f"模型架构: {architecture_name}, 输入尺寸: {img_size}, 类别数: {num_classes}")
    
    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    # 3. 打印模型结构，了解输入输出维度
    print(f"模型结构: {model}")
    
    # 查找分类器层
    classifier = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            classifier = module
            print(f"找到分类器层: {name}, 权重形状: {module.weight.shape}")
            break
    
    return model, img_size, num_classes, device, classifier

def get_classifier_layer(model):
    """提取模型的分类器层"""
    # 查找最后一个线性层作为分类器
    classifier = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            classifier = module
    return classifier