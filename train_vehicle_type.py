#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车型分类模型训练脚本（bus / car / minibus / truck）

解决当前 MiniVGGNet 分类混乱的问题：
  1. 输入尺寸从 100x100 提升到 224x224，保留更多细节
  2. 使用 ResNet18 替代 MiniVGGNet，特征提取能力更强
  3. 支持数据增强，提升泛化能力

数据格式要求：
  datasets/vehicle_type/
    train/
      bus/      <-- bus 类图片
      car/      <-- car 类图片
      minibus/  <-- minibus 类图片
      truck/    <-- truck 类图片
    val/
      bus/
      car/
      minibus/
      truck/
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import time
import copy

# ==================== 配置 ====================
DATA_DIR = 'datasets/vehicle_type'   # 数据集路径
NUM_CLASSES = 4                      # bus, car, minibus, truck
BATCH_SIZE = 32
NUM_EPOCHS = 50
INPUT_SIZE = 224                     # 提升输入分辨率
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = 'cfg/vehicle_type_v2.pth'

# ==================== 数据增强 ====================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==================== 加载数据 ====================
def load_data():
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"类别: {class_names}")
    print(f"训练集: {dataset_sizes['train']}, 验证集: {dataset_sizes['val']}")
    return dataloaders, dataset_sizes, class_names

# ==================== 构建模型 ====================
def build_model():
    # 使用预训练 ResNet18，替换最后的全连接层
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)
    return model

# ==================== 训练循环 ====================
def train_model(model, dataloaders, dataset_sizes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 40)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'*** 最佳模型更新，验证准确率: {best_acc:.4f} ***')
    
    print(f'\n训练完成！最佳验证准确率: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# ==================== 主函数 ====================
def main():
    print(f"使用设备: {DEVICE}")
    dataloaders, dataset_sizes, class_names = load_data()
    model = build_model()
    
    print("\n开始训练...")
    model = train_model(model, dataloaders, dataset_sizes)
    
    # 保存模型
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n模型已保存至: {SAVE_PATH}")
    
    # 同时保存类别映射
    with open('cfg/vehicle_type_classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
    print(f"类别映射已保存至: cfg/vehicle_type_classes.txt")

if __name__ == '__main__':
    main()
