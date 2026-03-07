import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import os
import numpy as np
from PIL import Image
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from typing import List, Tuple, Optional, Dict
import ssl
import requests
import warnings
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clean_images_and_paths(model_dir: str, num_images: int = 4) -> Tuple[List[torch.Tensor], List[str]]:
    """加载干净图像和对应的路径"""
    clean_data_dir = os.path.join(model_dir, "clean-example-data")
    
    if not os.path.exists(clean_data_dir):
        print(f"⚠️ 干净示例数据目录不存在: {clean_data_dir}")
        return [], []
    
    # 获取图像文件
    image_files = []
    for f in sorted(os.listdir(clean_data_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(clean_data_dir, f))
    
    if not image_files:
        print("⚠️ 未找到干净示例图像")
        return [], []
    
    # 加载并预处理图像
    images = []
    valid_paths = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    for i, img_path in enumerate(image_files[:num_images]):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor_img = transform(img).unsqueeze(0).to(device)  # [1, 3, 256, 256]
            images.append(tensor_img)
            valid_paths.append(img_path)
            print(f"✅ 加载图像 {i+1}: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"❌ 加载图像失败 {img_path}: {e}")
    
    return images, valid_paths