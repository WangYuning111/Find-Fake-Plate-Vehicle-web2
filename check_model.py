#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前模型配置，了解品牌类别和训练数据情况
"""
from ultralytics import YOLO
import torch
from conv import MiniVGGNet

print("=" * 60)
print("【1】YOLOv12 车辆检测/品牌识别模型")
print("=" * 60)

yolo = YOLO('cfg/best.pt')
print(f"模型类别数: {len(yolo.names)}")
print(f"类别映射: {yolo.names}")

print()
print("=" * 60)
print("【2】MiniVGGNet 车型分类模型")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v_type = MiniVGGNet(100, 100, 3, 4).to(device)
v_type.load_state_dict(torch.load("cfg/vehicle_type.pth", map_location=device))
print(f"输入尺寸: 100x100")
print(f"输出类别: bus, car, minibus, truck (共4类)")
print(f"模型参数量: {sum(p.numel() for p in v_type.parameters()):,}")

print()
print("=" * 60)
print("【3】MiniVGGNet 颜色分类模型")
print("=" * 60)

v_color = MiniVGGNet(100, 100, 3, 8).to(device)
v_color.load_state_dict(torch.load("cfg/vehicle_color.pth", map_location=device))
print(f"输入尺寸: 100x100")
print(f"输出类别: black, blue, brown, green, red, silver, white, yellow (共8色)")
print(f"模型参数量: {sum(p.numel() for p in v_color.parameters()):,}")

print()
print("=" * 60)
print("诊断结论")
print("=" * 60)
print("""
【品牌识别不准原因】
1. YOLO 检测模型同时承担了"定位车辆"和"识别品牌"两个任务
2. 目标检测模型对细粒度品牌分类能力天然有限（更关注位置而非车标细节）
3. 从仅有的品牌修正映射表看，训练时可能存在标签错误

【车型分类混乱原因】
1. MiniVGGNet 网络较浅（仅2个卷积块），表达能力有限
2. 输入尺寸仅100x100，车辆细节特征丢失严重
3. bus/minibus 外观相似度高，4分类可能训练数据不足
""")
