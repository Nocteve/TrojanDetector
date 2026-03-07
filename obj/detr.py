import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import json
import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import warnings
import matplotlib.pyplot as plt

# GLIDE 相关导入
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
import ssl
import requests

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- 自定义 DETR 模型定义（使用标准 ResNet backbone）--------------------
class MLP(nn.Module):
    """多层感知机，用于边界框回归头"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    手工实现的 DETR 模型，使用标准 ResNet 作为 backbone。
    backbone 应为一个标准的 resnet50 模型（来自 torchvision）。
    """
    def __init__(self, backbone: nn.Module, num_classes: int, num_queries: int = 100, aux_loss: bool = True):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        # 输入映射：将 backbone 输出的 2048 维特征映射到 transformer 的 256 维
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )

        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, 256)

        # 位置编码（可学习简化版）
        self.pos_embed = nn.Embedding(100, 256)

        # 分类头和边界框头
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 背景类
        self.bbox_embed = MLP(256, 256, 4, 3)

    def forward(self, images: torch.Tensor):
        """
        输入: images [batch, 3, H, W]
        输出: 字典包含 pred_logits 和 pred_boxes
        """
        # 1. 通过 backbone 提取特征（手动调用各层，直到 layer4）
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)      # [batch, 2048, H/32, W/32]

        batch, _, h, w = x.shape

        # 2. 投影到 transformer 维度
        src = self.input_proj(x)          # [batch, 256, h, w]

        # 3. 生成位置编码
        pos = self.pos_embed.weight[:h*w]               # [h*w, 256]
        pos = pos.view(h, w, 256).permute(2, 0, 1).unsqueeze(0)  # [1, 256, h, w]
        pos = pos.repeat(batch, 1, 1, 1)                 # [batch, 256, h, w]

        # 4. 展平为序列
        src = src.flatten(2).permute(2, 0, 1)            # [h*w, batch, 256]
        pos = pos.flatten(2).permute(2, 0, 1)            # [h*w, batch, 256]
        src = src + pos

        # 5. 准备查询
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)  # [num_queries, batch, 256]
        tgt = torch.zeros_like(query_embed)

        # 6. Transformer 前向
        memory = self.transformer.encoder(src)           # [h*w, batch, 256]
        hs = self.transformer.decoder(tgt, memory)       # [num_queries, batch, 256]

        # 7. 预测分类和边界框
        outputs_class = self.class_embed(hs)             # [num_queries, batch, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()    # [num_queries, batch, 4]

        # 8. 整理输出格式（batch first）
        out = {
            'pred_logits': outputs_class.permute(1, 0, 2),  # [batch, num_queries, num_classes+1]
            'pred_boxes': outputs_coord.permute(1, 0, 2),   # [batch, num_queries, 4]
        }

        if self.aux_loss:
            # 辅助输出（随便给几个假数据，不影响推理）
            out['aux_outputs'] = [
                {'pred_logits': out['pred_logits'], 'pred_boxes': out['pred_boxes']}
                for _ in range(5)
            ]
        return out


# -------------------- 后门检测器类（包含所有方法）--------------------
class DETRBackdoorDetector:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 检测器初始化，使用设备: {self.device}")

    def load_model(self, model_path: str, config_path: str) -> nn.Module:
        """加载 DETR 模型（改进的键名匹配，适配 conv_encoder 前缀和 coonv1 笔误）"""
        print(f"📦 加载模型权重: {model_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_classes = config.get('number_classes', 16)
        print(f"🔢 类别数: {num_classes}（+1 背景类）")

        # 1. 创建标准 ResNet50 作为 backbone
        backbone = resnet50(pretrained=False)
        backbone.eval()  # 暂时 eval，后面整体 eval

        # 2. 加载 state dict 并处理键名
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            # 处理可能的封装格式
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']

            # 创建新的 state_dict 用于 backbone
            backbone_state_dict = {}
            for k, v in state_dict.items():
                # 移除前缀 "conv_encoder.model."
                if k.startswith('conv_encoder.model.'):
                    new_k = k[19:]  # 去掉 "conv_encoder.model."
                else:
                    continue  # 忽略其他前缀，只处理 backbone 部分

                # 修正拼写错误：将 "coonv1" 替换为 "conv1"
                if new_k.startswith('coonv1'):
                    new_k = 'conv1' + new_k[6:]  # 替换前6个字符
                # 其他键名应该已经匹配标准 ResNet 的命名（如 layer1.0.conv1.weight）

                backbone_state_dict[new_k] = v

            # 3. 加载到 backbone
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing:
                print(f"⚠️ backbone 缺失键（可能为 fc 层等，不影响特征提取）: {missing[:10]}...")
            if unexpected:
                print(f"⚠️ backbone 意外键: {unexpected[:10]}...")
            print("✅ backbone 权重加载完成")

        except Exception as e:
            print(f"❌ 加载 state dict 失败: {e}")
            return None

        # 4. 创建 DETR 模型，传入 backbone
        model = DETR(backbone=backbone, num_classes=num_classes, num_queries=100, aux_loss=True)
        model.to(self.device)

        # 5. 加载 DETR 其他部分的权重（如果有的话），这里假设只有 backbone 有权重，其他部分随机初始化
        # 如果需要加载完整的 DETR 权重（包括 transformer 等），需另外处理

        model.eval()
        return model

    def load_annotations(self, image_path: str) -> Optional[List[Dict]]:
        """加载图像对应的标注文件（JSON 格式）"""
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
        """分析类别相似度（基于分类头权重）"""
        print("🔍 分析类别相似度...")
        weight = model.class_embed.weight.data  # [num_classes+1, 256]
        norm_w = weight / weight.norm(dim=1, keepdim=True)
        sim_matrix = norm_w @ norm_w.t()
        diag_mask = torch.eye(weight.shape[0], dtype=torch.bool, device=sim_matrix.device)
        sim_matrix.masked_fill_(diag_mask, float('inf'))
        results = []
        for i in range(1, weight.shape[0]):  # 跳过背景类 0
            valid_indices = list(range(1, weight.shape[0]))
            if i in valid_indices:
                valid_indices.remove(i)
            if not valid_indices:
                continue
            sim_values = sim_matrix[i, valid_indices]
            min_idx = torch.argmin(sim_values).item()
            j = valid_indices[min_idx]
            sim_val = sim_values[min_idx].item()
            results.append((i, j, sim_val))
        results.sort(key=lambda x: x[2])
        print(f"✅ 找到 {len(results)} 个类别对")
        for i, (src, tgt, sim) in enumerate(results[:5]):
            print(f"   {i+1}. 源{src}→目标{tgt}: 相似度={sim:.4f}")
        return results

    def get_detector_logits_direct(self, detection_model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """获取图像级 logits（取所有查询的最大值）"""
        with torch.set_grad_enabled(True):
            try:
                image = image.to(self.device)
                if image.shape[2] != 256 or image.shape[3] != 256:
                    image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
                outputs = detection_model(image)
                pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes+1]
                img_logits, _ = pred_logits.max(dim=1)  # [batch, num_classes+1]
                return img_logits
            except Exception as e:
                print(f"❌ 获取 DETR logits 失败: {e}")
                return torch.randn((image.size(0), 17), device=self.device, requires_grad=True)

    def setup_glide_model(self, timestep=50):
        """设置 GLIDE 扩散模型"""
        options = model_and_diffusion_defaults()
        options["timestep_respacing"] = str(timestep)
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if torch.cuda.is_available():
            model.convert_to_fp16()
        model.to(self.device)

        try:
            cache_dir = "./models_for_generating_triggers"
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_path = os.path.join(cache_dir, "base.pt")
            if not os.path.exists(checkpoint_path):
                print("正在下载 GLIDE 检查点...")
                url = "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt"
                response = requests.get(url, verify=False, timeout=30)
                with open(checkpoint_path, 'wb') as f:
                    f.write(response.content)
                print("GLIDE 检查点下载完成")
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print("✅ GLIDE 模型加载成功")
        except Exception as e:
            print(f"警告: 加载 GLIDE 检查点失败: {e}")
            print("使用随机权重 - 这可能会影响触发器生成质量")
        return model, diffusion, options

    def generate_trigger_with_classifier(
        self,
        detection_model: nn.Module,
        timestep: int,
        guidance_scale: float,
        target_label: int,
        target_image: torch.Tensor,
        source_label: int,
        add_noise: bool = True,
        grad_scale_factor: float = 1 / 7,
        noise_scale: float = 0.05
    ) -> Tuple[torch.Tensor, float]:
        """使用分类器引导的扩散模型生成加性噪声触发器"""
        print(f"🔄 生成加性噪声触发器: 源{source_label}→目标{target_label}, 噪声尺度: {noise_scale}")

        glide_model, diffusion, options = self.setup_glide_model(timestep)
        target_image = target_image.to(self.device)

        batch_size = 1
        prompt = " "
        tokens = glide_model.tokenizer.encode(prompt)
        tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
        model_kwargs = {
            "tokens": torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device),
            "mask": torch.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=torch.bool, device=self.device),
        }

        def cond_fn(x: torch.Tensor, t: torch.Tensor, y=None, **kwargs) -> torch.Tensor:
            with torch.enable_grad():
                x_single = x[0:1].clone().requires_grad_(True)

                if x_single.shape[2] != 256 or x_single.shape[3] != 256:
                    x0 = F.interpolate(x_single, size=(256, 256), mode='bilinear', align_corners=False)
                else:
                    x0 = x_single
                x0 = x0.requires_grad_(True)

                if add_noise:
                    input_img = x0 + 0.3 * torch.randn_like(x0).to(self.device)
                else:
                    input_img = x0
                input_img = input_img.clamp(0, 1)

                try:
                    logits = self.get_detector_logits_direct(detection_model, input_img)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)

                    probs = F.softmax(logits, dim=1)
                    eps = 1e-10
                    target_prob = probs[:, target_label] + eps
                    source_prob = probs[:, source_label] + eps

                    loss = torch.log(source_prob).mean() - torch.log(target_prob).mean()

                    grad = torch.autograd.grad(loss, x_single, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if grad is None or torch.isnan(grad).any() or torch.isinf(grad).any():
                        return torch.zeros_like(x_single)
                except Exception as e:
                    print(f"梯度计算失败: {e}")
                    return torch.zeros_like(x_single)

                l1_loss = x_single.abs().mean() * 0.1
                grad_l1 = torch.autograd.grad(l1_loss, x_single, retain_graph=True, create_graph=False, allow_unused=True)[0]
                if grad_l1 is None:
                    grad_l1 = torch.zeros_like(x_single)

                result = guidance_scale * grad - guidance_scale * grad_l1 * grad_scale_factor
                max_grad_norm = 0.05
                if result.norm() > max_grad_norm:
                    result = result * (max_grad_norm / result.norm())
                return result.expand(x.shape[0], -1, -1, -1)

        def model_fn(x_t, ts, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
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

        init_size = options["image_size"]
        if target_image.shape[2] != init_size or target_image.shape[3] != init_size:
            init_img = F.interpolate(target_image, size=(init_size, init_size), mode='bilinear', align_corners=False)
        else:
            init_img = target_image

        init_noise = noise_scale * torch.randn_like(init_img).expand(batch_size * 2, -1, -1, -1)

        glide_model.del_cache()
        try:
            print(f"开始扩散采样，时间步: {timestep}，引导强度: {guidance_scale}")
            samples = diffusion.p_sample_loop(
                model_fn,
                (batch_size * 2, 3, init_size, init_size),
                noise=init_noise,
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
            default_trigger = torch.randn_like(target_image) * noise_scale
            return default_trigger, 0.0

        if samples.shape[2] != target_image.shape[2] or samples.shape[3] != target_image.shape[3]:
            trigger_resized = F.interpolate(samples, size=target_image.shape[2:], mode='bilinear', align_corners=False)
        else:
            trigger_resized = samples

        trigger_scale = 0.2
        trigger_scaled = trigger_resized * trigger_scale
        trigger_mean = trigger_scaled.mean(dim=[1, 2, 3], keepdim=True)
        trigger_zero_mean = trigger_scaled - trigger_mean
        max_trigger_val = 0.20
        trigger_clamped = trigger_zero_mean.clamp(-max_trigger_val, max_trigger_val)

        if trigger_clamped.shape[2] > 32 and trigger_clamped.shape[3] > 32:
            kernel_size = 3
            sigma = 0.5
            channels = trigger_clamped.shape[1]
            x = torch.arange(kernel_size).float() - kernel_size // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            gauss_2d = torch.outer(gauss, gauss).view(1, 1, kernel_size, kernel_size)
            gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size).to(self.device)
            trigger_smooth = F.conv2d(trigger_clamped, gauss_2d, padding=kernel_size//2, groups=channels)
            trigger_final = trigger_smooth
        else:
            trigger_final = trigger_clamped

        with torch.no_grad():
            try:
                test_img = (target_image + trigger_final).clamp(0, 1)
                outputs = detection_model(test_img)
                pred_logits = outputs['pred_logits']
                probs = F.softmax(pred_logits, dim=-1)
                target_probs = probs[0, :, target_label]
                prob_target = target_probs.max().item() if target_probs.numel() > 0 else 0.0

                source_probs = probs[0, :, source_label]
                prob_source = source_probs.max().item() if source_probs.numel() > 0 else 0.0

                print(f"✅ 加性噪声触发器生成完成")
                print(f"   目标类别置信度: {prob_target:.4f}, 源类别置信度: {prob_source:.4f}")
                print(f"   触发器统计: 均值={trigger_final.mean().item():.4f}, 标准差={trigger_final.std().item():.4f}")
                print(f"   触发器范围: [{trigger_final.min().item():.4f}, {trigger_final.max().item():.4f}]")
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
        """评估触发器效果"""
        print(f"📊 评估触发器效果，目标类别: {target_label}")

        trigger = trigger.to(self.device)
        clean_images = clean_images.to(self.device)

        if clean_images.shape[2] != 256 or clean_images.shape[3] != 256:
            clean_images_resized = F.interpolate(clean_images, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            clean_images_resized = clean_images

        if trigger.shape[2] != 256 or trigger.shape[3] != 256:
            trigger_resized = F.interpolate(trigger, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            trigger_resized = trigger

        if trigger_resized.shape[0] == 1 and clean_images_resized.shape[0] > 1:
            trigger_resized = trigger_resized.expand(clean_images_resized.shape[0], -1, -1, -1)

        blended = (clean_images_resized + trigger_resized).clamp(0, 1)

        all_confidences = []
        with torch.no_grad():
            for i in range(blended.shape[0]):
                try:
                    image_tensor = blended[i].unsqueeze(0)
                    outputs = detection_model(image_tensor)
                    pred_logits = outputs['pred_logits']
                    probs = F.softmax(pred_logits, dim=-1)
                    target_probs = probs[0, :, target_label]
                    if target_probs.numel() > 0:
                        confidence = target_probs.max().item()
                        all_confidences.append(confidence)
                        print(f"  图像{i+1}: 目标类别最高置信度: {confidence:.4f}")
                except Exception as e:
                    print(f"  图像{i+1}检测失败: {e}")
                    continue

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
        guidance_scale: float = 50.0,
        num_iterations: int = 2,
        timestep: int = 50,
        noise_scale: float = 0.05
    ) -> Tuple[float, Optional[torch.Tensor], Dict]:
        """核心检测函数"""
        print("=" * 60)
        print("🔍 开始后门检测（DETR + GLIDE）")
        print("=" * 60)

        detection_model = detection_model.to(self.device)

        similarity_list = self.find_far(detection_model)
        candidate_pairs = similarity_list[:min(3, len(similarity_list))]
        print(f"候选类别对: {candidate_pairs}")

        score_list = []
        trigger_list = []
        pair_info_list = []

        for target_label, source_label, similarity in candidate_pairs:
            print(f"\n🔄 处理类别对: 源={source_label} → 目标={target_label} (相似度={similarity:.4f})")

            source_patch = None
            for img_idx, (img, img_path) in enumerate(zip(clean_images, image_paths)):
                annotations = self.load_annotations(img_path)
                if not annotations:
                    continue
                for ann in annotations:
                    if isinstance(ann, dict) and 'label' in ann and 'bbox' in ann:
                        if ann['label'] == source_label:
                            bbox = ann['bbox']
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                            if x2 > x1 and y2 > y1 and x2 <= img.shape[3] and y2 <= img.shape[2]:
                                patch = img[:, :, int(y1):int(y2), int(x1):int(x2)]
                                if patch.shape[2] > 0 and patch.shape[3] > 0:
                                    patch_resized = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
                                    source_patch = patch_resized
                                    print(f"✅ 从标注中找到源类别样本，边界框: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                                    break
                    if source_patch is not None:
                        break
                if source_patch is not None:
                    break

            if source_patch is None:
                print(f"⚠️ 未找到源类别 {source_label} 的标注样本，使用第一张图像")
                source_patch = clean_images[0]
                if source_patch.shape[2] != 64 or source_patch.shape[3] != 64:
                    source_patch = F.interpolate(source_patch, size=(64, 64), mode='bilinear', align_corners=False)

            current_guidance = guidance_scale
            current_noise = noise_scale
            for iteration in range(num_iterations):
                print(f"  迭代 {iteration+1}/{num_iterations}, 引导强度: {current_guidance}, 噪声尺度: {current_noise}")

                trigger, trigger_confidence = self.generate_trigger_with_classifier(
                    detection_model, timestep, current_guidance,
                    target_label, source_patch, source_label,
                    add_noise=True,
                    grad_scale_factor=1/7,
                    noise_scale=current_noise
                )

                stacked_images = torch.cat(clean_images, dim=0)
                score = self.evaluate_trigger_effect(detection_model, trigger, stacked_images, target_label)

                score_list.append(score)
                trigger_list.append(trigger)
                pair_info_list.append({
                    'source': source_label,
                    'target': target_label,
                    'similarity': similarity,
                    'trigger_confidence': trigger_confidence,
                    'iteration': iteration + 1,
                    'score': score,
                    'noise_scale': current_noise,
                    'guidance_scale': current_guidance
                })

                print(f"  ✅ 评估完成，后门得分: {score:.4f}")

                if score > 0.5:
                    break
                else:
                    current_guidance *= 1.2
                    current_noise *= 0.8

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