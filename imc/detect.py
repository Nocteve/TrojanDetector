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
'''
完整后门检测流程
'''
def detect_backdoor_in_classification_model(model_dir, clean_images_dir, 
                                           guidance_scale=100.0, num_iterations=2, 
                                           timestep=50, search_strategy="greedy"):
    """
    完整后门检测流程
    
    参数:
        model_dir: 模型目录
        clean_images_dir: 干净图像目录
        guidance_scale: 扩散引导强度
        num_iterations: 迭代次数
        timestep: 扩散步数
        search_strategy: 搜索策略 ('greedy' 或 'exhaustive')
    
    返回:
        max_score: 后门得分
        best_trigger: 最佳触发器
        best_pair: 最佳类别对
    """
    print("=" * 60)
    print("开始后门检测流程")
    print("=" * 60)
    
    # 1. 加载模型
    model, img_size, num_classes, device, classifier = load_classification_model(model_dir)
    
    print(f"✅ 模型加载完成，设备: {device}")
    
    if classifier is None:
        print("❌ 未找到分类器层")
        return 0.0, None, None
    
    print(f"✅ 分类器特征维度: {classifier.weight.shape[1]}")
    
    # 2. 加载干净图像
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    
    images = []
    labels = []
    
    for filename in os.listdir(clean_images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 从文件名解析标签，例如: class_4_example_19.png
            try:
                # 处理不同格式的文件名
                if 'class_' in filename:
                    label = int(filename.split('_')[1])
                elif 'example_' in filename:
                    # 尝试从其他格式解析
                    parts = filename.split('_')
                    for part in parts:
                        if part.isdigit():
                            label = int(part)
                            break
                    else:
                        continue
                else:
                    continue
                    
                img_path = os.path.join(clean_images_dir, filename)
                image = Image.open(img_path).convert('RGB')
                if isinstance(image,Image.Image):
                    tensor = transform(image).to(device)
                    #print(tensor)  tensor已经归一化过了
                    images.append(tensor)
                    labels.append(label)
            except (IndexError, ValueError):
                continue
    
    if not images:
        print("❌ 未找到有效图像")
        return 0.0, None, None
    
    images = torch.stack(images)  # [N, C, H, W]
    labels = torch.tensor(labels, device=device)
    
    print(f"✅ 加载 {len(images)} 张干净图像")
    print(f"✅ 图像形状: {images.shape}")
    
    # 3. 分析类别相似度
    print("\n📊 正在分析类别相似度...")
    similarity_list = find_far(classifier)
    print(f"✅ 找到 {len(similarity_list)} 个类别对")
    
    # 显示类别对
    for i, (target, source, sim) in enumerate(similarity_list):
        print(f"  类别对 {i+1}: 目标={target}, 源={source}, 相似度={sim:.4f}")
    
    # 4. 根据搜索策略选择类别对
    if search_strategy == "greedy":
        candidate_pairs = similarity_list
        print("✅ 使用贪心搜索策略")
    elif search_strategy == "exhaustive":
        # 生成所有可能的类别对
        candidate_pairs = []
        for target_label in range(num_classes):
            for source_label in range(num_classes):
                if target_label != source_label:
                    candidate_pairs.append((target_label, source_label, 0.0))
        print(f"✅ 使用穷举搜索策略，共 {len(candidate_pairs)} 个类别对")
    else:
        raise ValueError(f"未知搜索策略: {search_strategy}")
    
    # 5. 为每个候选对生成触发器并评估
    print("\n🔬 开始生成触发器并评估...")
    score_list = []
    trigger_list = []
    pair_list = []
    
    for pair_idx, (target_label, source_label, similarity) in enumerate(candidate_pairs):
        print(f"\n  处理类别对 {pair_idx+1}/{len(candidate_pairs)}: "
              f"目标={target_label}, 源={source_label}")
        
        # 获取源和目标类别的样本
        target_mask = labels == target_label
        source_mask = labels == source_label
        
        target_images = images[target_mask]
        source_images = images[source_mask]
        
        if len(target_images) == 0:
            print(f"  警告: 缺少目标类别 {target_label} 的样本，跳过")
            continue
            
        if len(source_images) == 0:
            print(f"  警告: 缺少源类别 {source_label} 的样本，跳过")
            continue
        
        print(f"  目标类别样本数: {len(target_images)}, 源类别样本数: {len(source_images)}")
        
        # 使用第一个样本生成触发器
        target_image = target_images[0:1]
        
        # 生成触发器
        current_guidance_scale = guidance_scale
        trigger = None
        trigger_confidence = 0.0
        
        for iteration in range(num_iterations):
            print(f"    迭代 {iteration+1}: guidance_scale={current_guidance_scale}")
            
            try:
                trigger, trigger_confidence = generate_trigger_with_classifier(
                    model, classifier, timestep, current_guidance_scale, target_label,
                    target_image, source_label, device=device
                )
                
                if trigger_confidence > 0.95 or iteration == num_iterations - 1:
                    print(f"    ✅ 触发器生成成功，置信度: {trigger_confidence:.4f}")
                    break
                else:
                    current_guidance_scale *= 1.5
                    print(f"    ⚠️  置信度不足 {trigger_confidence:.4f}，增加guidance_scale")
            except Exception as e:
                print(f"    ❌ 触发器生成失败: {e}")
                break
        
        if trigger is not None:
            # 评估触发器
            try:
                score = evaluate_trigger(model, trigger, images, target_label)
                import math 
                if not math.isnan(score):
                    score_list.append(score)
                    trigger_list.append(trigger)
                    pair_list.append((target_label, source_label))
                
                print(f"    📈 触发器评估得分: {score:.4f}")
            except Exception as e:
                print(f"    ❌ 触发器评估失败: {e}")
    
    # 6. 分析结果
    print("\n" + "=" * 60)
    print("检测结果分析")
    print("=" * 60)
    
    if score_list:
        max_idx = np.argmax(score_list)
        max_score = score_list[max_idx]
        best_trigger = trigger_list[max_idx]
        best_pair = pair_list[max_idx]
        
        print(f"✅ 检测完成!")
        print(f"📊 最佳类别对: 目标={best_pair[0]}, 源={best_pair[1]}")
        print(f"📈 最高后门得分: {max_score:.4f}")
        
        # 判断是否后门
        threshold = 0.5
        if max_score > threshold:
            print(f"🚨 检测结果: 模型可能存在后门!")
            print(f"   后门可能将类别 {best_pair[1]} 重定向到类别 {best_pair[0]}")
        else:
            print(f"✅ 检测结果: 模型可能安全")
        
        return max_score, best_trigger, best_pair
    else:
        print("⚠️  未生成有效触发器，无法进行检测")
        return 0.0, None, None