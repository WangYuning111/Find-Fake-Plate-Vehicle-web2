#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆品牌独立分类模型训练脚本

核心改进：将"品牌识别"从 YOLO 检测中解耦出来
  - YOLO 只负责：定位车辆位置（检测框）
  - 本模型负责：对检测框内的车辆图像进行品牌分类

优势：
  1. 检测模型不再需要学习品牌细粒度特征
  2. 品牌分类模型可以专门优化，输入更大、网络更深
  3. 新增品牌只需重新训练本模型，无需重新标注检测数据

数据格式要求（ImageFolder格式）：
  datasets/vehicle_brand_cls/
    train/
      大众/      <-- 大众品牌车辆图片
      宝马/
      丰田/
      本田/
      奔驰/
      奥迪/
      别克/
      ...（其他品牌）
    val/
      大众/
      宝马/
      ...

建议收集图片时：
  - 优先收集车辆前部（车标清晰）
  - 其次收集整车侧面/后部
  - 每个品牌至少 200-500 张
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
import time
import copy

# ==================== 配置 ====================
DATA_DIR = 'datasets/vehicle_brand_cls'   # 品牌分类数据集路径
BATCH_SIZE = 32
NUM_EPOCHS = 80                           # 品牌分类更难，需要更多轮数
INPUT_SIZE = 224
LR = 0.0005                               # 使用预训练权重，学习率稍低
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = 'cfg/vehicle_brand_cls.pth'

# ==================== 数据增强（品牌识别对细节要求高） ====================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"数据集目录不存在: {DATA_DIR}\n"
            f"请按以下结构准备数据:\n"
            f"  {DATA_DIR}/\n"
            f"    train/\n"
            f"      大众/ (*.jpg)\n"
            f"      宝马/ (*.jpg)\n"
            f"      ...\n"
            f"    val/\n"
            f"      大众/ (*.jpg)\n"
            f"      ..."
        )
    
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
    
    print(f"=" * 60)
    print(f"品牌类别 ({len(class_names)} 类): {class_names}")
    print(f"训练集: {dataset_sizes['train']} 张")
    print(f"验证集: {dataset_sizes['val']} 张")
    print(f"=" * 60)
    
    return dataloaders, dataset_sizes, class_names

# ==================== 构建模型（ResNet50 更深，细粒度分类更好） ====================
def build_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    
    # 添加 Dropout 防止过拟合
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model = model.to(DEVICE)
    return model

# ==================== 训练循环 ====================
def train_model(model, dataloaders, dataset_sizes, num_classes):
    criterion = nn.CrossEntropyLoss()
    
    # 微调策略：前面层学习率小，后面层学习率大
    # 冻结前两层（layer1, layer2），只训练后面层和全连接层
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # 为不同层设置不同学习率
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': LR * 0.1},
        {'params': model.layer4.parameters(), 'lr': LR * 0.5},
        {'params': model.fc.parameters(), 'lr': LR}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()}')
        print('-' * 60)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                # 训练时解冻所有层（后期微调）
                if epoch == 20:
                    print("[INFO] Epoch 20: 解冻 layer2")
                    for param in model.layer2.parameters():
                        param.requires_grad = True
                if epoch == 40:
                    print("[INFO] Epoch 40: 解冻 layer1")
                    for param in model.layer1.parameters():
                        param.requires_grad = True
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
            
            print(f'{phase:5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'*** 最佳模型更新: Val Acc = {best_acc:.4f} ***')
    
    print(f'\n{"=" * 60}')
    print(f'训练完成！最佳验证准确率: {best_acc:.4f}')
    print(f'{"=" * 60}')
    model.load_state_dict(best_model_wts)
    return model

# ==================== 主函数 ====================
def main():
    print(f"使用设备: {DEVICE}")
    dataloaders, dataset_sizes, class_names = load_data()
    
    model = build_model(len(class_names))
    print(f"\n模型: ResNet50，输出类别数: {len(class_names)}")
    
    model = train_model(model, dataloaders, dataset_sizes, len(class_names))
    
    # 保存模型
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n模型权重已保存: {SAVE_PATH}")
    
    # 保存类别映射（用于推理时解码）
    import json
    brand_map = {i: name for i, name in enumerate(class_names)}
    with open('cfg/vehicle_brand_classes.json', 'w', encoding='utf-8') as f:
        json.dump(brand_map, f, ensure_ascii=False, indent=2)
    print(f"类别映射已保存: cfg/vehicle_brand_classes.json")
    
    # 导出为 TorchScript（可选，加速推理）
    # model.eval()
    # example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    # traced = torch.jit.trace(model, example)
    # traced.save('cfg/vehicle_brand_cls_traced.pt')
    # print("TorchScript 模型已导出")

if __name__ == '__main__':
    main()
