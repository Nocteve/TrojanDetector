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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 修复SSL验证问题
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FasterRCNNBackdoorDetector:
    """Faster R-CNN后门检测器 - 修复维度问题和触发器置信度计算"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 检测器初始化，使用设备: {self.device}")
        
    def load_model(self, model_path: str, config_path: str) -> nn.Module:
        """加载Faster R-CNN模型"""
        print(f"📦 加载模型: {model_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载模型
        try:
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model = model.to(self.device)
            model.eval()
            print(f"✅ 模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def load_annotations(self, image_path: str) -> Optional[List[Dict]]:
        """加载图像对应的标注文件"""
        json_path = image_path.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')
        
        if not os.path.exists(json_path):
            return None
            
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        except Exception as e:
            print(f"⚠️ 加载标注失败 {json_path}: {e}")
            return None
    
    def find_far(self, model: nn.Module) -> List[Tuple[int, int, float]]:
        """
        贪心算法：为每个类别找到最不相似的类别
        基于Faster R-CNN的box_predictor权重计算余弦相似度
        """
        print("🔍 分析类别相似度...")
        
        # 获取分类器权重
        predictor = model.roi_heads.box_predictor
        if hasattr(predictor, 'cls_score'):
            weight = predictor.cls_score.weight.data  # [num_classes, feature_dim]
            print(f"📊 权重形状: {weight.shape}")
        else:
            raise ValueError("无法找到分类器权重")
        
        num_classes = weight.shape[0]  # 包括背景类
        
        # 计算余弦相似度矩阵
        norm_w = weight / weight.norm(dim=1, keepdim=True)
        sim_matrix = norm_w @ norm_w.t()  # [num_classes, num_classes]
        
        # 掩码对角线（自身相似度）
        diag_mask = torch.eye(num_classes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix.masked_fill_(diag_mask, float('inf'))
        
        # 为每个类别找到最不相似的类别（排除背景类0）
        results = []
        for i in range(1, num_classes):  # 从1开始，跳过背景类
            # 找到最不相似的类别（排除自身）
            valid_indices = list(range(1, num_classes))
            if i in valid_indices:
                valid_indices.remove(i)
            
            if valid_indices:
                sim_values = sim_matrix[i, valid_indices]
                min_idx = torch.argmin(sim_values).item()
                j = valid_indices[min_idx]
                sim_val = sim_values[min_idx].item()
                results.append((i, j, sim_val))
        
        # 按相似度排序（最不相似的在前）
        results.sort(key=lambda x: x[2])
        print(f"✅ 找到 {len(results)} 个类别对")
        for i, (src, tgt, sim) in enumerate(results[:5]):
            print(f"   {i+1}. 源{src}→目标{tgt}: 相似度={sim:.4f}")
        
        return results
    
    def setup_glide_model(self, timestep=50):
        """设置GLIDE扩散模型"""
        options = model_and_diffusion_defaults()
        options["timestep_respacing"] = str(timestep)
        
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        
        if torch.cuda.is_available():
            model.convert_to_fp16()
        
        model.to(self.device)
        
        try:
            # 尝试从本地缓存加载
            cache_dir = "./models_for_generating_triggers"
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_path = os.path.join(cache_dir, "base.pt")
            
            if not os.path.exists(checkpoint_path):
                print("正在下载GLIDE检查点...")
                # 使用requests下载
                url = "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt"
                response = requests.get(url, verify=False, timeout=30)
                with open(checkpoint_path, 'wb') as f:
                    f.write(response.content)
                print("GLIDE检查点下载完成")
            
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print("✅ GLIDE模型加载成功")
            
        except Exception as e:
            print(f"警告: 加载GLIDE检查点失败: {e}")
            print("使用随机权重 - 这可能会影响触发器生成质量")
        
        return model, diffusion, options
    
    def get_detector_logits_direct(self, detection_model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        直接获取检测器的logits - 修复维度问题
        关键：确保特征正确传递到box_head
        """
        with torch.set_grad_enabled(True):
            try:
                # 确保图像在正确的设备上
                image = image.to(self.device)
                
                # 获取backbone特征
                features_dict = detection_model.backbone(image)
                
                # 使用最高层特征
                if isinstance(features_dict, dict):
                    if '3' in features_dict:
                        features = features_dict['3']
                    elif 'pool' in features_dict:
                        features = features_dict['pool']
                    else:
                        last_key = list(features_dict.keys())[-1]
                        features = features_dict[last_key]
                else:
                    features = features_dict
                
                # 关键：确保特征维度正确 [batch, channels, height, width]
                # 自适应池化到7x7
                features_pooled = F.adaptive_avg_pool2d(features, (7, 7))
                
                # 检查box_head的结构
                box_head = detection_model.roi_heads.box_head
                
                # 尝试不同的输入方式
                try:
                    # 方式1：保持4D格式 [batch, channels, height, width]
                    box_features = box_head(features_pooled)
                except Exception as e1:
                    try:
                        # 方式2：展平为2D [batch, features]
                        print(f"⚠️ box_head 4D输入失败，尝试展平: {e1}")
                        batch_size = features_pooled.size(0)
                        # 展平但保持批次维度
                        features_flat = features_pooled.view(batch_size, -1)
                        box_features = box_head(features_flat)
                    except Exception as e2:
                        # 方式3：使用备用网络
                        print(f"⚠️ box_head 2D输入也失败，使用备用网络: {e2}")
                        # 全局平均池化
                        global_pooled = F.adaptive_avg_pool2d(features_pooled, (1, 1))
                        global_flat = global_pooled.view(global_pooled.size(0), -1)
                        
                        # 创建或使用备用网络
                        if not hasattr(self, 'backup_network'):
                            in_features = global_flat.shape[1]
                            out_features = 1024  # 假设输出维度
                            self.backup_network = nn.Sequential(
                                nn.Linear(in_features, out_features),
                                nn.ReLU(),
                                nn.Linear(out_features, out_features),
                                nn.ReLU()
                            ).to(self.device)
                            # 初始化
                            for layer in self.backup_network:
                                if isinstance(layer, nn.Linear):
                                    nn.init.normal_(layer.weight, std=0.01)
                                    nn.init.zeros_(layer.bias)
                        
                        box_features = self.backup_network(global_flat)
                
                # 获取logits
                box_predictor = detection_model.roi_heads.box_predictor
                logits = box_predictor(box_features)
                
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                return logits
                
            except Exception as e:
                print(f"❌ 获取检测器logits失败: {e}")
                # 返回随机logits以允许梯度计算
                return torch.randn((image.size(0), 17), device=self.device, requires_grad=True)
    
    def generate_trigger_with_classifier(
        self, 
        detection_model: nn.Module,
        timestep: int,
        guidance_scale: float,
        target_label: int,
        target_image: torch.Tensor,
        source_label: int,
        add_noise: bool = True,
        grad_scale_factor: float = 1/7,
        noise_scale: float = 0.05  # 控制噪声尺度，使其成为小扰动
    ) -> Tuple[torch.Tensor, float]:
        """
        使用分类器引导的扩散模型生成加性噪声触发器
        关键修改：从小噪声开始，生成小的加性扰动
        """
        print(f"🔄 生成加性噪声触发器: 源{source_label}→目标{target_label}, 噪声尺度: {noise_scale}")
        
        # 1. 设置GLIDE扩散模型
        glide_model, diffusion, options = self.setup_glide_model(timestep)
        
        # 确保目标图像在正确的设备上
        target_image = target_image.to(self.device)
        
        # 2. 准备扩散模型输入 
        batch_size = 1
        prompt = " "
        tokens = glide_model.tokenizer.encode(prompt)
        tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
        
        model_kwargs = {
            "tokens": torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device),
            "mask": torch.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=torch.bool, device=self.device),
        }
        
        # 3. 定义条件函数（核心引导逻辑）
        def cond_fn(x: torch.Tensor, t: torch.Tensor, y=None, **kwargs) -> torch.Tensor:
            """
            修正的cond_fn函数签名，生成加性噪声触发器
            """
            with torch.enable_grad():
                # 提取当前图像
                x_single = x[0:1].clone().requires_grad_(True)
                
                # 预处理图像 - 调整到256x256
                if x_single.shape[2] != 256 or x_single.shape[3] != 256:
                    x0 = F.interpolate(x_single, size=(256, 256), mode='bilinear', align_corners=False)
                else:
                    x0 = x_single
                
                x0 = x0.requires_grad_(True)
                
                # 添加噪声增强鲁棒性
                if add_noise:
                    noise_level = 0.3  # 较大的噪声以获得更好的鲁棒性
                    input_img = x0 + noise_level * torch.randn_like(x0).to(self.device)
                else:
                    input_img = x0
                
                input_img = input_img.clamp(0, 1)
                
                # 获取分类logits
                try:
                    # 使用修复后的函数获取logits
                    logits = self.get_detector_logits_direct(detection_model, input_img)
                    
                    # 确保logits形状正确
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    
                    # 确保logits需要梯度
                    if not logits.requires_grad:
                        logits.requires_grad_(True)
                    
                    # 核心梯度计算：最大化目标类别，最小化源类别
                    probs = F.softmax(logits, dim=1)
                    
                    # 添加微小值避免数值问题
                    eps = 1e-10
                    target_prob = probs[:, target_label] + eps
                    source_prob = probs[:, source_label] + eps
                    
                    # 损失 = log(p(source)) - log(p(target))
                    loss = torch.log(source_prob).mean() - torch.log(target_prob).mean()
                    
                    # 计算梯度
                    grad = torch.autograd.grad(
                        loss, x_single, 
                        retain_graph=True, 
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    if grad is None:
                        print("⚠️ 梯度为None，返回零梯度")
                        return torch.zeros_like(x_single)
                    
                    # 检查梯度是否有效
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print("⚠️ 梯度包含NaN或Inf，返回零梯度")
                        return torch.zeros_like(x_single)
                        
                except Exception as e:
                    print(f"梯度计算失败: {e}")
                    return torch.zeros_like(x_single)
                
                # L1正则化：控制触发器大小（更强调加性噪声特性）
                l1_loss = x_single.abs().mean() * 0.1  # 增加正则化强度以鼓励稀疏性
                
                try:
                    grad_l1 = torch.autograd.grad(
                        l1_loss, x_single, 
                        retain_graph=True, 
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    if grad_l1 is None:
                        grad_l1 = torch.zeros_like(x_single)
                        
                except Exception as e:
                    grad_l1 = torch.zeros_like(x_single)
                
                # 总梯度 = 引导梯度 - L1正则化
                adjusted_guidance_scale = guidance_scale
                result = adjusted_guidance_scale * grad - adjusted_guidance_scale * grad_l1 * grad_scale_factor
                
                # 梯度裁剪，防止爆炸
                max_grad_norm = 0.05
                grad_norm = result.norm()
                if grad_norm > max_grad_norm:
                    result = result * (max_grad_norm / grad_norm)
                
                # 扩展到batch大小
                result = result.expand(x.shape[0], -1, -1, -1)
                
                return result
        
        # 4. 定义无分类器引导的模型函数
        def model_fn(x_t, ts, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            
            # 确保传递正确的参数
            model_kwargs_filtered = {}
            if 'tokens' in kwargs:
                model_kwargs_filtered['tokens'] = kwargs['tokens']
            if 'mask' in kwargs:
                model_kwargs_filtered['mask'] = kwargs['mask']
            
            model_out = glide_model(combined, ts, **model_kwargs_filtered)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        
        # 5. 准备初始噪声（从小噪声开始）
        init_size = options["image_size"]
        
        if target_image.shape[2] != init_size or target_image.shape[3] != init_size:
            init_img = F.interpolate(target_image, size=(init_size, init_size), 
                                   mode='bilinear', align_corners=False)
        else:
            init_img = target_image
        
        # 关键修改：从小随机噪声开始，而不是零噪声或目标图像
        # 这有助于生成加性噪声触发器
        init_noise = noise_scale * torch.randn_like(init_img).expand(batch_size * 2, -1, -1, -1)
        
        # 6. 运行扩散采样
        glide_model.del_cache()
        try:
            print(f"开始扩散采样，时间步: {timestep}，引导强度: {guidance_scale}")
            
            # 使用DDIM采样
            samples = diffusion.p_sample_loop(
                model_fn,
                (batch_size * 2, 3, init_size, init_size),
                noise=init_noise,  # 从小噪声开始
                device=self.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )[:batch_size]
            
            glide_model.del_cache()
            print("扩散采样完成")
            
        except Exception as e:
            print(f"❌ 扩散采样失败: {e}")
            glide_model.del_cache()
            # 返回一个小的随机触发器作为加性噪声
            default_trigger = torch.randn_like(target_image) * noise_scale
            return default_trigger, 0.0
        
        # 7. 调整触发器大小回原始尺寸
        if samples.shape[2] != target_image.shape[2] or samples.shape[3] != target_image.shape[3]:
            trigger_resized = F.interpolate(samples, size=target_image.shape[2:], 
                                          mode='bilinear', align_corners=False)
        else:
            trigger_resized = samples
        
        # 关键修改：生成加性噪声触发器（而不是覆盖性触发器）
        # 1. 缩放触发器，使其成为小扰动
        trigger_scale = 0.2  # 控制触发器幅度
        trigger_scaled = trigger_resized * trigger_scale
        
        # 2. 确保触发器均值接近0，保持加性性质
        trigger_mean = trigger_scaled.mean(dim=[1, 2, 3], keepdim=True)
        trigger_zero_mean = trigger_scaled - trigger_mean
        
        # 3. 限制触发器的绝对大小，确保是小扰动
        max_trigger_val = 0.20  # 最大扰动幅度
        trigger_clamped = trigger_zero_mean.clamp(-max_trigger_val, max_trigger_val)
        
        # 4. 可选：应用轻微的高斯模糊使触发器更平滑
        if trigger_clamped.shape[2] > 32 and trigger_clamped.shape[3] > 32:
            # 创建高斯模糊核
            kernel_size = 3
            sigma = 0.5
            channels = trigger_clamped.shape[1]
            
            # 创建一维高斯核
            x = torch.arange(kernel_size).float() - kernel_size // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            
            # 创建2D高斯核
            gauss_2d = torch.outer(gauss, gauss)
            gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
            gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
            gauss_2d = gauss_2d.to(self.device)
            
            # 应用高斯模糊
            trigger_smooth = F.conv2d(
                trigger_clamped, 
                gauss_2d, 
                padding=kernel_size//2, 
                groups=channels
            )
            trigger_final = trigger_smooth
        else:
            trigger_final = trigger_clamped
        
        # 8. 计算触发器置信度 - 修复：计算触发器叠加后的效果
        with torch.no_grad():
            try:
                # 创建带触发器的图像
                test_img = (target_image + trigger_final).clamp(0, 1)
                
                # 运行检测器
                detection_model.eval()
                image_tensor = test_img.squeeze(0)  # [3, H, W]
                
                # 运行检测
                predictions = detection_model([image_tensor])
                pred = predictions[0]
                
                # 计算目标类别的最高置信度
                if len(pred['labels']) > 0:
                    target_indices = pred['labels'] == target_label
                    if target_indices.any():
                        prob_target = pred['scores'][target_indices].max().item()
                    else:
                        prob_target = 0.0
                else:
                    prob_target = 0.0
                
                # 同时计算源类别的置信度（用于调试）
                if len(pred['labels']) > 0:
                    source_indices = pred['labels'] == source_label
                    if source_indices.any():
                        prob_source = pred['scores'][source_indices].max().item()
                    else:
                        prob_source = 0.0
                else:
                    prob_source = 0.0
                
                # 检查触发器特性
                trigger_mean_val = trigger_final.mean().item()
                trigger_std_val = trigger_final.std().item()
                trigger_min_val = trigger_final.min().item()
                trigger_max_val = trigger_final.max().item()
                
                print(f"✅ 加性噪声触发器生成完成")
                print(f"   目标类别置信度: {prob_target:.4f}, 源类别置信度: {prob_source:.4f}")
                print(f"   触发器统计: 均值={trigger_mean_val:.4f}, 标准差={trigger_std_val:.4f}")
                print(f"   触发器范围: [{trigger_min_val:.4f}, {trigger_max_val:.4f}]")
                
                return trigger_final, prob_target
                
            except Exception as e:
                print(f"⚠️ 置信度计算失败: {e}")
                return trigger_final, 0.0
    
    def evaluate_trigger_effect(
        self, 
        detection_model: nn.Module,
        trigger: torch.Tensor,
        clean_images: torch.Tensor,
        target_label: int
    ) -> float:
        """
        评估触发器效果：将触发器应用到干净图像上，测量模型对目标类别的平均响应
        注意：触发器应该是加性的小扰动
        """
        print(f"📊 评估触发器效果，目标类别: {target_label}")
        
        # 确保所有数据在正确的设备上
        trigger = trigger.to(self.device)
        clean_images = clean_images.to(self.device)
        
        # 1. 调整触发器大小以匹配图像
        if clean_images.shape[2] != 256 or clean_images.shape[3] != 256:
            clean_images_resized = F.interpolate(clean_images, size=(256, 256), 
                                               mode='bilinear', align_corners=False)
        else:
            clean_images_resized = clean_images
        
        if trigger.shape[2] != 256 or trigger.shape[3] != 256:
            trigger_resized = F.interpolate(trigger, size=(256, 256), 
                                          mode='bilinear', align_corners=False)
        else:
            trigger_resized = trigger
        
        # 2. 应用触发器到所有图像
        if trigger_resized.shape[0] == 1 and clean_images_resized.shape[0] > 1:
            trigger_resized = trigger_resized.expand(clean_images_resized.shape[0], -1, -1, -1)
        
        # 关键：确保触发器是加性的，并且范围适当
        blended = (clean_images_resized + trigger_resized).clamp(0, 1)
        
        # 3. 运行检测并计算目标类别的平均置信度
        all_confidences = []
        
        with torch.no_grad():
            for i in range(blended.shape[0]):
                # 单张图像检测
                try:
                    # Faster R-CNN期望3D张量
                    image_tensor = blended[i]  # 形状应该是 [3, H, W]
                    
                    predictions = detection_model([image_tensor])
                    pred = predictions[0]
                    
                    # 过滤检测结果，只保留目标类别
                    if len(pred['labels']) > 0 and len(pred['scores']) > 0:
                        target_indices = pred['labels'] == target_label
                        if target_indices.any():
                            target_confidences = pred['scores'][target_indices]
                            if len(target_confidences) > 0:
                                confidence = target_confidences.max().item()
                                all_confidences.append(confidence)
                                print(f"  图像{i+1}: 检测到目标类别，最高置信度: {confidence:.4f}")
                except Exception as e:
                    print(f"  图像{i+1}检测失败: {e}")
                    continue
        
        # 4. 计算平均置信度
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            print(f"📊 触发器评估: {len(all_confidences)}个检测，平均置信度: {avg_confidence:.4f}")
            return avg_confidence
        else:
            print("⚠️ 未检测到目标类别")
            return 0.0

    def detect_backdoor(
        self,
        detection_model: nn.Module,
        clean_images: List[torch.Tensor],
        image_paths: List[str],
        guidance_scale: float = 50.0,  # 增加引导强度
        num_iterations: int = 2,
        timestep: int = 50,  # 增加时间步以获得更好的质量
        noise_scale: float = 0.05  # 新增：控制初始噪声尺度
    ) -> Tuple[float, Optional[torch.Tensor], Dict]:
        """
        核心检测函数 - 使用完整GLIDE扩散模型
        """
        print("=" * 60)
        print("🔍 开始后门检测（使用GLIDE扩散模型）")
        print("=" * 60)
        
        # 确保模型在正确的设备上
        detection_model = detection_model.to(self.device)
        
        # 1. 分析类别相似度
        print("📊 分析类别相似度...")
        similarity_list = self.find_far(detection_model)
        candidate_pairs = similarity_list[:min(3, len(similarity_list))]
        print(f"候选类别对: {candidate_pairs}")
        
        # 2. 为每个候选对生成触发器
        score_list = []
        trigger_list = []
        pair_info_list = []
        
        for target_label, source_label, similarity in candidate_pairs:
            print(f"\n🔄 处理类别对: 源={source_label} → 目标={target_label} (相似度={similarity:.4f})")
            
            # 寻找源类别的样本（使用标注信息）
            source_patch = None
            
            for img_idx, (img, img_path) in enumerate(zip(clean_images, image_paths)):
                # 加载标注信息
                annotations = self.load_annotations(img_path)
                if not annotations:
                    continue
                
                # 在标注中查找源类别的边界框
                for ann in annotations:
                    if isinstance(ann, dict) and 'label' in ann and 'bbox' in ann:
                        if ann['label'] == source_label:
                            # 提取边界框
                            bbox = ann['bbox']
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                            
                            # 确保边界框有效
                            if x2 > x1 and y2 > y1 and x2 <= img.shape[3] and y2 <= img.shape[2]:
                                # 裁剪并调整大小
                                patch = img[:, :, int(y1):int(y2), int(x1):int(x2)]
                                if patch.shape[2] > 0 and patch.shape[3] > 0:
                                    # 调整到64x64用于扩散模型
                                    patch_resized = F.interpolate(patch, size=(64, 64), 
                                                                 mode='bilinear', align_corners=False)
                                    source_patch = patch_resized
                                    print(f"✅ 从标注中找到源类别样本，边界框: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                                    break
                
                if source_patch is not None:
                    break
            
            if source_patch is None:
                print(f"⚠️ 未找到源类别 {source_label} 的标注样本，使用第一张图像")
                # 使用第一张干净图像作为基础
                source_patch = clean_images[0]
                if source_patch.shape[2] != 64 or source_patch.shape[3] != 64:
                    source_patch = F.interpolate(source_patch, size=(64, 64), 
                                                mode='bilinear', align_corners=False)
            
            # 生成触发器 - 使用完整的GLIDE扩散模型
            current_guidance_scale = guidance_scale
            current_noise_scale = noise_scale
            
            for iteration in range(num_iterations):
                print(f"  迭代 {iteration+1}/{num_iterations}, 引导强度: {current_guidance_scale}, 噪声尺度: {current_noise_scale}")
                
                # 使用完整的GLIDE扩散模型生成加性噪声触发器
                trigger, trigger_confidence = self.generate_trigger_with_classifier(
                    detection_model, timestep, current_guidance_scale,
                    target_label, source_patch, source_label,
                    add_noise=True,  # 保持噪声
                    grad_scale_factor=1/7,
                    noise_scale=current_noise_scale  # 传入噪声尺度
                )
                
                print(f"  触发器置信度: {trigger_confidence:.4f}")
                
                # 评估触发器
                stacked_images = torch.cat(clean_images, dim=0)
                score = self.evaluate_trigger_effect(detection_model, trigger, 
                                                   stacked_images, target_label)
                
                score_list.append(score)
                trigger_list.append(trigger)
                pair_info_list.append({
                    'source': source_label,
                    'target': target_label,
                    'similarity': similarity,
                    'trigger_confidence': trigger_confidence,
                    'iteration': iteration + 1,
                    'score': score,
                    'noise_scale': current_noise_scale,
                    'guidance_scale': current_guidance_scale
                })
                
                print(f"  ✅ 评估完成，后门得分: {score:.4f}")
                
                # 如果得分较高，可以提前停止
                if score > 0.5:
                    break
                else:
                    # 调整引导强度和噪声尺度
                    current_guidance_scale *= 1.2
                    current_noise_scale *= 0.8  # 减小噪声尺度以获得更精细的触发器
        
        # 3. 返回最佳结果
        if score_list:
            max_idx = np.argmax(score_list)
            max_score = score_list[max_idx]
            best_trigger = trigger_list[max_idx]
            best_pair_info = pair_info_list[max_idx]
            
            print(f"\n🏆 最佳结果:")
            print(f"   源类别: {best_pair_info['source']} → 目标类别: {best_pair_info['target']}")
            print(f"   相似度: {best_pair_info['similarity']:.4f}")
            print(f"   触发器置信度: {best_pair_info['trigger_confidence']:.4f}")
            print(f"   后门得分: {max_score:.4f}")
            
            # 检查触发器的特性
            print(f"   触发器统计: 均值={best_trigger.mean().item():.4f}, 标准差={best_trigger.std().item():.4f}")
            print(f"   触发器范围: [{best_trigger.min().item():.4f}, {best_trigger.max().item():.4f}]")
            
            results = {
                'max_score': max_score,
                'best_trigger': best_trigger,
                'pair_info': best_pair_info,
                'all_scores': score_list,
                'all_pair_info': pair_info_list
            }
            
            return max_score, best_trigger, results
        else:
            print("\n❌ 未生成有效的触发器")
            return 0.0, None, {}