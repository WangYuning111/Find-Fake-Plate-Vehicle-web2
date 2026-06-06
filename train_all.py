#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端模型训练脚本
- 自动从反馈数据库构建训练集
- 数据增强
- Fine-tune 车型分类 + 颜色分类模型
- 支持增量训练（在现有模型基础上继续训练）

用法:
    python train_all.py --task type    # 训练车型分类模型
    python train_all.py --task color   # 训练颜色分类模型
    python train_all.py --task all     # 训练所有模型
"""
import os
import sys
import argparse
import csv
import shutil
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from conv import MiniVGGNet
from preprocessing import AspectAwarePreprocessor, ImageToTensorPreprocessor
from improve_accuracy import augment_image

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[TRAIN] 使用设备: {DEVICE}")


class VehicleDataset(Dataset):
    """车辆分类数据集"""
    def __init__(self, image_paths, labels, classes, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.augment = augment
        self.aap = AspectAwarePreprocessor(100, 100)
        self.iap = ImageToTensorPreprocessor(data_format='channels_first')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            # 如果图片读取失败，返回一个黑色图
            img = np.zeros((100, 100, 3), dtype=np.uint8)

        if self.augment:
            # 对重点类别增加更多增强变体
            if label in ['white', 'silver']:
                num_variants = 6
            elif label in ['brown', 'red', 'bus']:
                num_variants = 5
            else:
                num_variants = 2
            aug_imgs = augment_image(img, num_variants=num_variants)
            img = aug_imgs[-1] if len(aug_imgs) > 1 else img

        roi = self.aap.preprocess(img)
        tensor = self.iap.preprocess(roi)
        target = self.class_to_idx.get(label, 0)
        return tensor, target


# 本地已分类数据文件夹映射
LOCAL_TYPE_FOLDERS = {'bus', 'car', 'minibus', 'truck'}
LOCAL_COLOR_FOLDERS = {'black', 'blue', 'brown', 'green', 'red', 'silver', 'white', 'yellow'}


def load_feedback_data(task='type'):
    """
    从反馈数据库加载训练数据
    :param task: 'type' 或 'color'
    :return: (image_paths, labels, classes)
    """
    feedback_csv = os.path.join(os.path.dirname(Config.DATABASE_PATH), "feedback.csv")
    if not os.path.exists(feedback_csv):
        print(f"[TRAIN] 反馈数据库不存在: {feedback_csv}")
        return None, None, None

    image_paths = []
    labels = []

    with open(feedback_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'corrected':
                continue
            img_path = row.get('image_path', '')
            if not os.path.exists(img_path):
                continue

            if task == 'type':
                label = row.get('corrected_type') or row.get('v_type', '')
            elif task == 'color':
                label = row.get('corrected_color') or row.get('v_color', '')
            else:
                continue

            if not label or label == 'unknown':
                continue

            image_paths.append(img_path)
            labels.append(label)

    if len(image_paths) < 10:
        print(f"[TRAIN] 修正数据不足 ({len(image_paths)} 条)，需要至少 10 条才能训练")
        return None, None, None

    # 统计类别
    classes = sorted(set(labels))
    print(f"[TRAIN] 加载 {len(image_paths)} 条 {task} 数据，共 {len(classes)} 个类别: {classes}")
    return image_paths, labels, classes


def load_local_data(src_dir, task='type'):
    """
    从本地已按文件夹分类的图片加载训练数据
    :param src_dir: 图片根目录，子目录名为标签
    :param task: 'type' 或 'color'
    :return: (image_paths, labels, classes)
    """
    if not os.path.isdir(src_dir):
        print(f"[TRAIN] 数据目录不存在: {src_dir}")
        return None, None, None

    if task == 'type':
        valid_folders = LOCAL_TYPE_FOLDERS
    elif task == 'color':
        valid_folders = LOCAL_COLOR_FOLDERS
    else:
        print(f"[TRAIN] 未知任务: {task}")
        return None, None, None

    image_paths = []
    labels = []

    for folder_name in sorted(os.listdir(src_dir)):
        folder_path = os.path.join(src_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.lower() not in valid_folders:
            continue

        label = folder_name.lower()
        count = 0
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                image_paths.append(os.path.join(folder_path, fname))
                labels.append(label)
                count += 1
        print(f"[TRAIN]   类别 '{label}': {count} 张")

    if len(image_paths) < 10:
        print(f"[TRAIN] 数据不足 ({len(image_paths)} 张)，需要至少 10 张才能训练")
        return None, None, None

    classes = sorted(set(labels))
    print(f"[TRAIN] 从本地加载 {len(image_paths)} 张 {task} 图片，共 {len(classes)} 个类别: {classes}")
    return image_paths, labels, classes


def train_model(task, image_paths, labels, classes, epochs=50, batch_size=16, lr=0.001):
    """训练模型"""
    num_classes = len(classes)
    print(f"[TRAIN] 开始训练 {task} 分类模型，类别数: {num_classes}")

    # 划分训练/验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 统计各类别数量，计算类别权重（解决类别不平衡）
    from collections import Counter
    label_counts = Counter(labels)
    print(f"[TRAIN] 类别分布: {dict(label_counts)}")
    
    # 计算权重: 总样本数 / (类别数 * 该类样本数)
    total = len(labels)
    n_classes = len(classes)
    class_weights = {}
    for cls in classes:
        count = label_counts.get(cls, 1)
        class_weights[cls] = total / (n_classes * count)
    
    # 对重点改进类别增加额外权重
    if task == 'color':
        class_weights['brown'] = class_weights.get('brown', 1.0) * 3.0  # brown 重点加强
        class_weights['red'] = class_weights.get('red', 1.0) * 2.5  # red 加强
        class_weights['silver'] = class_weights.get('silver', 1.0) * 1.5
        class_weights['white'] = class_weights.get('white', 1.0) * 2.0
        class_weights['blue'] = class_weights.get('blue', 1.0) * 1.5
    elif task == 'type':
        class_weights['bus'] = class_weights.get('bus', 1.0) * 2.0  # bus 重点加强
    
    print(f"[TRAIN] 类别权重: {class_weights}")
    
    # 数据集
    train_dataset = VehicleDataset(train_paths, train_labels, classes, augment=True)
    val_dataset = VehicleDataset(val_paths, val_labels, classes, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型
    model = MiniVGGNet(100, 100, 3, num_classes).to(DEVICE)

    # 尝试加载预训练权重（增量训练）
    pretrained_path = None
    if task == 'type':
        pretrained_path = Config.VEHICLE_TYPE_MODEL_PATH
    elif task == 'color':
        pretrained_path = Config.VEHICLE_COLOR_MODEL_PATH

    if pretrained_path and os.path.exists(pretrained_path):
        try:
            # 加载预训练权重，但忽略最后一层（类别数可能不同）
            pretrained_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
            model_dict = model.state_dict()
            # 过滤掉最后一层
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"[TRAIN] 已加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"[TRAIN] 预训练权重加载失败: {e}，将从头训练")

    # 使用类别权重构建损失函数（仅用于损失计算，不改变采样）
    weight_tensor = torch.tensor([class_weights.get(c, 1.0) for c in classes], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr*0.5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    best_val_acc = 0.0
    best_model_path = None
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_acc = 100. * train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_acc:.2f}%  Val Loss: {val_loss/len(val_loader):.4f}  "
              f"Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f"cfg/vehicle_{task}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.2f}%) 到 {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  -> 验证准确率连续 {max_patience} 轮未提升，提前停止训练")
                break

    # 同时保存最终模型
    final_path = f"cfg/vehicle_{task}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[TRAIN] 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"[TRAIN] 最佳模型: {best_model_path}")
    print(f"[TRAIN] 最终模型: {final_path}")
    return best_model_path


def auto_train_all(min_samples=20, src_dir=None):
    """
    自动训练所有模型
    :param min_samples: 每个任务最少需要的样本数
    :param src_dir: 本地已分类数据目录，None 则使用反馈数据库
    """
    print("="*60)
    print("[AUTO-TRAIN] 自动训练流程启动")
    if src_dir:
        print(f"[AUTO-TRAIN] 数据来源: {src_dir}")
    else:
        print("[AUTO-TRAIN] 数据来源: 反馈数据库")
    print("="*60)

    # 训练车型分类
    if src_dir:
        type_paths, type_labels, type_classes = load_local_data(src_dir, 'type')
    else:
        type_paths, type_labels, type_classes = load_feedback_data('type')
    if type_paths and len(type_paths) >= min_samples:
        print("\n" + "="*60)
        train_model('type', type_paths, type_labels, type_classes, epochs=30)
    else:
        print(f"\n[SKIP] 车型分类数据不足 (需要 {min_samples} 条)")

    # 训练颜色分类
    if src_dir:
        color_paths, color_labels, color_classes = load_local_data(src_dir, 'color')
    else:
        color_paths, color_labels, color_classes = load_feedback_data('color')
    if color_paths and len(color_paths) >= min_samples:
        print("\n" + "="*60)
        train_model('color', color_paths, color_labels, color_classes, epochs=30)
    else:
        print(f"\n[SKIP] 颜色分类数据不足 (需要 {min_samples} 条)")

    print("\n" + "="*60)
    print("[AUTO-TRAIN] 训练流程结束")
    print("提示: 将训练好的模型文件重命名为 cfg/vehicle_type.pth 和 cfg/vehicle_color.pth 即可生效")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='车辆分类模型训练')
    parser.add_argument('--task', choices=['type', 'color', 'all'], default='all',
                        help='训练任务: type=车型, color=颜色, all=全部')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--min-samples', type=int, default=10, help='最少样本数')
    parser.add_argument('--src', type=str, default=None,
                        help='本地已分类数据目录（子目录名为标签）')
    args = parser.parse_args()

    if args.task == 'all':
        auto_train_all(min_samples=args.min_samples, src_dir=args.src)
    else:
        if args.src:
            paths, labels, classes = load_local_data(args.src, args.task)
        else:
            paths, labels, classes = load_feedback_data(args.task)
        if paths and len(paths) >= args.min_samples:
            train_model(args.task, paths, labels, classes,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        else:
            print(f"[ERR] 数据不足，需要至少 {args.min_samples} 条修正数据")
            print("请先在系统中上传图片并修正标签，或运行: python batch_collect.py --src <图片文件夹>")
