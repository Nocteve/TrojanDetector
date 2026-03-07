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

from obj.fasterRcnn import FasterRCNNBackdoorDetector
from obj.load_images import load_clean_images_and_paths


def visualize_trigger(trigger: torch.Tensor, save_path: str = None):
    """可视化触发器"""
    import matplotlib.pyplot as plt
    
    t = trigger.squeeze(0).cpu()
    trigger_np=t
    t = t.mean(0, keepdim=True).repeat(3,1,1) if t.size(0)>3 else t  # 多通道转3通道
    t = t.permute(1,2,0).numpy()
    trigger_min,trigger_max=t.min(),t.max()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # 归一化
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(t if t.shape[-1]==3 else t.squeeze(-1), cmap='gray')
    
    plt.title(f'触发器 (范围: [{trigger_min:.3f}, {trigger_max:.3f}])')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # 显示触发器直方图
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
    """主函数"""
    # 设置路径
    model_dir = "E:\\code\\XINAN\\obj_test\\id-00000002"
    config_path = os.path.join(model_dir, "reduced-config.json")
    model_path = os.path.join(model_dir, "model.pt")
    
    # 检查路径是否存在
    print(f"📁 检查路径:")
    print(f"  模型目录: {model_dir} - {'✅ 存在' if os.path.exists(model_dir) else '❌ 不存在'}")
    print(f"  配置文件: {config_path} - {'✅ 存在' if os.path.exists(config_path) else '❌ 不存在'}")
    print(f"  模型文件: {model_path} - {'✅ 存在' if os.path.exists(model_path) else '❌ 不存在'}")
    
    # 初始化检测器
    detector = FasterRCNNBackdoorDetector(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    print("\n📦 加载目标检测模型...")
    model = detector.load_model(model_path, config_path)
    if model is None:
        print("❌ 模型加载失败，退出程序")
        return
    
    # 2. 加载干净图像和路径
    print("\n🖼️ 加载干净示例图像...")
    clean_images, image_paths = load_clean_images_and_paths(model_dir, num_images=4)
    if not clean_images:
        print("❌ 无法加载干净图像，退出")
        return
    
    # 3. 运行后门检测
    print("\n🔬 运行后门检测算法（使用GLIDE扩散模型）...")
    score, trigger, results = detector.detect_backdoor(
        model, clean_images, image_paths,
        guidance_scale=80.0,
        num_iterations=2,
        timestep=50,  # 增加时间步以获得更好的质量
        noise_scale=0.2  # 控制初始噪声尺度
    )
    
    # 4. 输出结果
    print("\n" + "=" * 60)
    print("📋 检测结果")
    print("=" * 60)
    
    if score > 0.45:
        print(f"🚨 警告: 检测到潜在后门!")
        print(f"   后门得分: {score:.4f} (阈值: 0.45)")
        
        if trigger is not None:
            output_dir = os.path.join(model_dir, "backdoor_detection")
            os.makedirs(output_dir, exist_ok=True)
            
            # 可视化触发器
            trigger_viz_path = os.path.join(output_dir, "trigger_visualization.png")
            visualize_trigger(trigger, trigger_viz_path)
            
            # 保存触发器为图像
            try:
                # trigger_pic = transforms.ToPILImage()(trigger.squeeze(0).cpu())
                # 核心：先处理多通道+归一化，再转PIL图像
                t = trigger.squeeze(0).cpu()
                # 64通道→3通道（均值压缩+扩RGB），≤3通道直接用
                t = t.mean(0, keepdim=True).repeat(3,1,1) if t.size(0)>3 else t
                # 归一化到0-255并转uint8
                t = (t - t.min()) / (t.max() - t.min() + 1e-8) * 255
                t = t.byte()
                # 生成正确的PIL图像
                trigger_pic = transforms.ToPILImage()(t)

                trigger_path = os.path.join(output_dir, "generated_trigger.png")
                trigger_pic.save(trigger_path)
                print(f"✅ 触发器已保存至: {trigger_path}")
            except Exception as e:
                print(f"⚠️ 保存触发器图像失败: {e}")
                
                # 备选方案：保存为numpy文件
                trigger_np = trigger.squeeze(0).cpu().numpy()
                trigger_path = os.path.join(output_dir, "generated_trigger.npy")
                np.save(trigger_path, trigger_np)
                print(f"✅ 触发器已保存至: {trigger_path}")
            
            # 保存详细结果
            results_path = os.path.join(output_dir, "detection_results.json")
            with open(results_path, 'w') as f:
                # 创建可序列化的结果字典
                serializable_results = {}
                for key, value in results.items():
                    if key == 'best_trigger':
                        continue  # 跳过张量
                    elif torch.is_tensor(value):
                        if value.numel() == 1:
                            serializable_results[key] = value.item()
                        else:
                            serializable_results[key] = value.cpu().tolist()
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2)
            print(f"✅ 详细结果已保存至: {results_path}")
    else:
        print(f"✅ 模型看起来是干净的")
        print(f"   后门得分: {score:.4f} (阈值: 0.3)")
    
    print("\n" + "=" * 60)
    print("🏁 检测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()