import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models_for_generating_triggers.glide import generate_trigger_with_classifier
from imc.load_model import load_classification_model,get_classifier_layer
from imc.similarity_analysis import find_far
from imc.evaluate import evaluate_trigger
from imc.detect import detect_backdoor_in_classification_model

# ==================== 6. 主程序 ====================
def main():
    """主程序"""
    # 设置路径
    model_dir = "E:\\code\\XINAN\\image-classification-jun2020-test\\models\\models\\id-00000001"
    clean_images_dir = os.path.join(model_dir, "clean-example-data")
    
    if not os.path.exists(clean_images_dir):
        clean_images_dir = os.path.join(model_dir, "example_data")
    
    print("🧪 DISTIL 后门检测系统")
    print("=" * 60)
    
    # 检查目录
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        return
    
    if not os.path.exists(clean_images_dir):
        print(f"错误: 干净图像目录不存在: {clean_images_dir}")
        return
    
    print(f"模型目录: {model_dir}")
    print(f"干净图像目录: {clean_images_dir}")
    
    # 运行后门检测
    try:
        score, trigger, best_pair = detect_backdoor_in_classification_model(
            model_dir=model_dir,
            clean_images_dir=clean_images_dir,
            guidance_scale=100.0,
            num_iterations=2,
            timestep=50,
            search_strategy="greedy"
        )#exhaustive or greedy

        # import math
        # if math.isnan(score):
        #     print(math.isinf(score))

        # 保存结果
        if trigger is not None:
            # 保存触发器图像
            trigger_path = os.path.join(model_dir, "detected_trigger.png")
            trigger_img = trigger.squeeze(0).cpu()
            trigger_img = transforms.ToPILImage()(trigger_img)
            trigger_img.save(trigger_path)
            print(f"\n💾 触发器已保存至: {trigger_path}")
            
            # 保存检测报告
            report_path = os.path.join(model_dir, "backdoor_detection_report.txt")
            with open(report_path, "w") as f:
                f.write("DISTIL 后门检测报告\n")
                f.write("=" * 40 + "\n")
                f.write(f"模型目录: {model_dir}\n")
                import datetime
                f.write(f"检测时间: {datetime.datetime.now()}\n")
                f.write(f"最佳类别对: 目标={best_pair[0]}, 源={best_pair[1]}\n")
                f.write(f"后门得分: {score:.4f}\n")
                f.write(f"检测结果: {'可能存在后门' if score > 0.6 else '可能安全'}\n")
                f.write("=" * 40 + "\n")
            
            print(f"📄 检测报告已保存至: {report_path}")
            
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()