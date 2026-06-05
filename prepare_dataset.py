#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集整理工具

功能：将下载的原始图片按类别整理为 ImageFolder 格式，
      供 train_vehicle_type.py 和 train_vehicle_brand.py 使用

用法示例：
  # 1. 整理车型分类数据
  python prepare_dataset.py \
      --src "D:/Downloads/UA-DETRAC/images" \
      --dst "datasets/vehicle_type" \
      --task type \
      --split 0.8

  # 2. 整理品牌分类数据（手动按品牌分好文件夹后）
  python prepare_dataset.py \
      --src "D:/Downloads/CompCars/sorted_by_brand" \
      --dst "datasets/vehicle_brand_cls" \
      --task brand \
      --split 0.8
"""
import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='整理车辆数据集')
    parser.add_argument('--src', required=True, help='原始图片目录')
    parser.add_argument('--dst', required=True, help='输出目录')
    parser.add_argument('--task', choices=['type', 'brand'], required=True,
                        help='任务类型: type=车型分类, brand=品牌分类')
    parser.add_argument('--split', type=float, default=0.8, help='训练集比例 (默认0.8)')
    parser.add_argument('--min_images', type=int, default=10,
                        help='每类最少图片数，少于则丢弃 (默认10)')
    parser.add_argument('--max_images', type=int, default=2000,
                        help='每类最多图片数，多于则随机采样 (默认2000)')
    return parser.parse_args()

def collect_images(src_dir):
    """收集所有图片文件，按文件夹（类别）分组"""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    classes = defaultdict(list)
    
    src_path = Path(src_dir)
    
    # 遍历第一层子文件夹作为类别
    for class_dir in src_path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.rglob('*'):
            if img_path.suffix.lower() in image_exts:
                classes[class_name].append(str(img_path))
    
    return classes

def split_and_copy(classes, dst_dir, split_ratio, min_images, max_images):
    """划分训练集/验证集并复制到目标目录"""
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    stats = []
    
    for class_name, images in sorted(classes.items()):
        # 过滤样本过少的类别
        if len(images) < min_images:
            print(f"[SKIP] 类别 '{class_name}' 样本数 {len(images)} < {min_images}，已跳过")
            continue
        
        # 随机采样（避免某类过多导致不平衡）
        if len(images) > max_images:
            random.shuffle(images)
            images = images[:max_images]
            print(f"[SAMPLE] 类别 '{class_name}' 样本过多，已采样至 {max_images} 张")
        
        # 随机划分
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # 创建类别目录
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # 复制文件
        for img in train_imgs:
            shutil.copy2(img, os.path.join(train_dir, class_name, os.path.basename(img)))
        for img in val_imgs:
            shutil.copy2(img, os.path.join(val_dir, class_name, os.path.basename(img)))
        
        stats.append((class_name, len(train_imgs), len(val_imgs)))
        print(f"[OK] {class_name}: 训练 {len(train_imgs)} | 验证 {len(val_imgs)}")
    
    return stats

def print_summary(stats, dst_dir):
    """输出整理结果摘要"""
    total_train = sum(s[1] for s in stats)
    total_val = sum(s[2] for s in stats)
    
    print("\n" + "=" * 60)
    print("数据集整理完成！")
    print("=" * 60)
    print(f"输出路径: {dst_dir}")
    print(f"类别数: {len(stats)}")
    print(f"训练集: {total_train} 张")
    print(f"验证集: {total_val} 张")
    print(f"总计: {total_train + total_val} 张")
    print("=" * 60)
    print("\n类别分布:")
    for name, train_cnt, val_cnt in stats:
        print(f"  {name:12} | 训练: {train_cnt:4} | 验证: {val_cnt:4}")

def main():
    args = parse_args()
    random.seed(42)
    
    print(f"正在整理数据集...")
    print(f"  源目录: {args.src}")
    print(f"  目标目录: {args.dst}")
    print(f"  任务: {args.task}")
    print(f"  训练集比例: {args.split}")
    
    # 收集图片
    classes = collect_images(args.src)
    if not classes:
        print(f"[ERROR] 在 {args.src} 下未找到任何图片")
        print("请确保目录结构为: 源目录/类别名/图片文件")
        return
    
    print(f"\n发现 {len(classes)} 个类别")
    
    # 划分并复制
    stats = split_and_copy(classes, args.dst, args.split, args.min_images, args.max_images)
    
    # 输出摘要
    print_summary(stats, args.dst)

if __name__ == '__main__':
    main()
