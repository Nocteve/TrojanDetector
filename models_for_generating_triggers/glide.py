import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.download import load_checkpoint
import requests
import ssl
'''
扩散模型引导生成
'''
# ==================== 修复SSL错误 ====================
# 禁用SSL验证（临时解决方案，仅用于开发）
ssl._create_default_https_context = ssl._create_unverified_context

def setup_glide_model(timestep=50, device=None):
    """设置GLIDE扩散模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    options = model_and_diffusion_defaults()
    options["use_fp16"] = device.type == "cuda"
    options["timestep_respacing"] = str(timestep)
    
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    
    if device.type == "cuda":
        model.convert_to_fp16()
    
    model.to(device)
    
    try:
        # 尝试从本地缓存加载
        cache_dir = os.path.expanduser("./models_for_generating_triggers")
        os.makedirs(cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(cache_dir, "base.pt")
        
        if not os.path.exists(checkpoint_path):
            print("正在下载GLIDE检查点...")
            # 手动下载
            url = "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt"
            response = requests.get(url, verify=False)  # verify=False 禁用SSL验证
            with open(checkpoint_path, 'wb') as f:
                f.write(response.content)
            print("GLIDE检查点下载完成")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    except Exception as e:
        print(f"警告: 加载GLIDE检查点失败: {e}")
        print("使用随机权重 - 这可能会影响触发器生成质量")
    
    return model, diffusion, options

def generate_trigger_with_classifier(full_model, classifier, timestep, guidance_scale, 
                                    target_label, target_image, source_label, 
                                    add_noise=True, grad_scale_factor=1/7, device=None):
    """
    使用分类器引导的扩散模型生成触发器
    核心方程: x_{t-1} = 扩散采样(x_t, t) + λ∇_x[log p(y=target|x) - log p(y=source|x)] - β∇_x||x||_1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 设置GLIDE模型
    glide_model, diffusion, options = setup_glide_model(timestep, device)
    
    # 2. 准备扩散模型输入
    batch_size = 1
    prompt = " "  # 空提示用于无分类器引导
    
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
    
    model_kwargs = {
        "tokens": torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
        "mask": torch.tensor([mask] * batch_size + [uncond_mask] * batch_size, 
                            dtype=torch.bool, device=device),
    }
    
    # 3. 定义条件函数（核心引导逻辑）
    def cond_fn(x, t, full_model, guidance_scale):
        with torch.enable_grad():
            # 提取当前图像 [batch_size * 2, 3, 64, 64]
            x_single = x[0:1].clone().requires_grad_(True)
            
            # 调整图像大小到模型输入尺寸
            # 注意：我们需要知道原始模型期望的输入大小
            # 假设模型期望224x224（从配置读取）
            x0 = F.interpolate(x_single, size=(224, 224), mode='bilinear', align_corners=False)
            x0 = x0.requires_grad_(True)
            
            # 添加噪声增强鲁棒性
            if add_noise:
                input_img = x0 + 0.3 * torch.rand_like(x0).to(device)
            else:
                input_img = x0
            
            # 前向传播获取分类logits - 使用整个模型
            logits = full_model(input_img.clamp(0, 1))
            
            # 确保logits是正确的形状
            if logits.dim() > 2:
                logits = logits.mean(dim=[2, 3])  # 对空间维度平均
            
            # 核心梯度计算：最大化目标类别，最小化源类别
            loss = (logits[:, target_label] - logits[:, source_label]).mean()
            grad_target = torch.autograd.grad(loss, x_single, retain_graph=True)[0]
            
            # L1正则化：控制触发器大小
            l1_loss = x_single.norm(p=1)
            grad_l1 = torch.autograd.grad(l1_loss, x_single, retain_graph=True)[0]
            
            guidance_scale=guidance_scale #避免过大出现nan?
            # 总梯度 = 引导梯度 - L1正则化
            return guidance_scale * grad_target - guidance_scale * grad_l1 * grad_scale_factor
    
    def cond_fn_wrapper(x, t, **kwargs):
        return cond_fn(x, t, full_model, guidance_scale)
    
    # 4. 定义无分类器引导的模型函数
    def model_fn(x_t, ts, **kwargs):
        half = x_t[:len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    # 5. 准备初始噪声（从目标图像开始）
    init_size = options["image_size"]  # GLIDE默认64
    print(f"目标图像形状: {target_image.shape}")
    print(f"GLIDE输入大小: {init_size}")
    
    # 调整目标图像到GLIDE输入大小
    target_image_64 = F.interpolate(target_image, size=(init_size, init_size), 
                                   mode="bilinear", align_corners=False)
    init_noise = target_image_64.expand(batch_size * 2, -1, -1, -1)
    
    # 6. 运行扩散采样
    glide_model.del_cache()
    print("开始扩散采样...")
    samples = diffusion.p_sample_loop(
        model_fn,
        (batch_size * 2, 3, init_size, init_size),
        noise=init_noise,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn_wrapper,
    )[:batch_size]
    glide_model.del_cache()
    
    # 7. 调整触发器大小回原始尺寸
    trigger = F.interpolate(samples, size=target_image.shape[2:], 
                           mode="bilinear", align_corners=False)
    
    # 8. 计算触发器置信度
    with torch.no_grad():
        logits = full_model(trigger.clamp(0, 1))
        if logits.dim() > 2:
            logits = logits.mean(dim=[2, 3])
        prob_target = F.softmax(logits, dim=1)[0, target_label].item()
    
    return trigger.clamp(0, 1), prob_target