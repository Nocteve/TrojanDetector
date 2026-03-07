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

from obj.detr import DETRBackdoorDetector

# -------------------- 辅助函数 --------------------
def load_clean_images_and_paths(model_dir: str, num_images: int = 4) -> Tuple[List[torch.Tensor], List[str]]:
    clean_data_dir = os.path.join(model_dir, "clean-example-data")
    if not os.path.exists(clean_data_dir):
        print(f"⚠️ 干净示例数据目录不存在: {clean_data_dir}")
        return [], []

    image_files = []
    for f in sorted(os.listdir(clean_data_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(clean_data_dir, f))

    if not image_files:
        return [], []

    images = []
    valid_paths = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for i, img_path in enumerate(image_files[:num_images]):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor_img = transform(img).unsqueeze(0).to(device)
            images.append(tensor_img)
            valid_paths.append(img_path)
            print(f"✅ 加载图像 {i+1}: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"❌ 加载图像失败 {img_path}: {e}")
    return images, valid_paths


def visualize_trigger(trigger: torch.Tensor, save_path: str = None):
    t = trigger.squeeze(0).cpu()
    trigger_np = t
    t = t.mean(0, keepdim=True).repeat(3, 1, 1) if t.size(0) > 3 else t
    t = t.permute(1, 2, 0).numpy()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(t if t.shape[-1] == 3 else t.squeeze(-1), cmap='gray')
    plt.title('触发器')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.hist(trigger_np.flatten(), bins=50)
    plt.title(f'触发器分布\n均值: {trigger_np.mean():.4f}, 标准差: {trigger_np.std():.4f}')
    plt.xlabel('值')
    plt.ylabel('频率')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 触发器可视化已保存至: {save_path}")
    plt.show()


def main():
    model_dir = "E:\\code\\XINAN\\obj_test\\id-00000004"  # 请根据实际情况修改
    config_path = os.path.join(model_dir, "reduced-config.json")
    model_path = os.path.join(model_dir, "model-state-dict.pt")

    print(f"📁 检查路径:")
    print(f"  模型目录: {model_dir} - {'✅ 存在' if os.path.exists(model_dir) else '❌ 不存在'}")
    print(f"  配置文件: {config_path} - {'✅ 存在' if os.path.exists(config_path) else '❌ 不存在'}")
    print(f"  模型文件: {model_path} - {'✅ 存在' if os.path.exists(model_path) else '❌ 不存在'}")

    detector = DETRBackdoorDetector(device='cuda' if torch.cuda.is_available() else 'cpu')

    print("\n📦 加载 DETR 模型...")
    model = detector.load_model(model_path, config_path)
    if model is None:
        print("❌ 模型加载失败，退出程序")
        return

    print("\n🖼️ 加载干净示例图像...")
    clean_images, image_paths = load_clean_images_and_paths(model_dir, num_images=4)
    if not clean_images:
        print("❌ 无法加载干净图像，退出")
        return

    print("\n🔬 运行后门检测算法（使用 GLIDE 扩散模型）...")
    score, trigger, results = detector.detect_backdoor(
        model, clean_images, image_paths,
        guidance_scale=80.0,
        num_iterations=2,
        timestep=50,
        noise_scale=0.2
    )

    print("\n" + "=" * 60)
    print("📋 检测结果")
    print("=" * 60)
    score*=10
    if score > 0.45:
        print(f"🚨 警告: 检测到潜在后门!")
        print(f"   后门得分: {score:.4f} (阈值: 0.45)")
        if trigger is not None:
            output_dir = os.path.join(model_dir, "backdoor_detection")
            os.makedirs(output_dir, exist_ok=True)
            trigger_viz_path = os.path.join(output_dir, "trigger_visualization.png")
            visualize_trigger(trigger, trigger_viz_path)

            try:
                t = trigger.squeeze(0).cpu()
                t = t.mean(0, keepdim=True).repeat(3, 1, 1) if t.size(0) > 3 else t
                t = (t - t.min()) / (t.max() - t.min() + 1e-8) * 255
                t = t.byte()
                trigger_pic = transforms.ToPILImage()(t)
                trigger_path = os.path.join(output_dir, "generated_trigger.png")
                trigger_pic.save(trigger_path)
                print(f"✅ 触发器已保存至: {trigger_path}")
            except Exception as e:
                print(f"⚠️ 保存触发器图像失败: {e}")
                trigger_np = trigger.squeeze(0).cpu().numpy()
                np.save(os.path.join(output_dir, "generated_trigger.npy"), trigger_np)
                print(f"✅ 触发器已保存至: {os.path.join(output_dir, 'generated_trigger.npy')}")

            with open(os.path.join(output_dir, "detection_results.json"), 'w') as f:
                serializable_results = {}
                for key, value in results.items():
                    if key == 'best_trigger':
                        continue
                    elif torch.is_tensor(value):
                        serializable_results[key] = value.cpu().tolist() if value.numel() > 1 else value.item()
                    else:
                        serializable_results[key] = value
                json.dump(serializable_results, f, indent=2)
            print(f"✅ 详细结果已保存至: {os.path.join(output_dir, 'detection_results.json')}")
    else:
        print(f"✅ 模型看起来是干净的")
        print(f"   后门得分: {score:.4f} (阈值: 0.45)")

    print("\n" + "=" * 60)
    print("🏁 检测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()