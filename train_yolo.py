#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12 车辆检测模型 Fine-tune 脚本
用途：在现有 best.pt 基础上，用更多品牌/车型数据继续训练

数据格式要求：
  datasets/
    vehicle_brand/
      images/
        train/  <-- 训练图片
        val/    <-- 验证图片
      labels/
        train/  <-- YOLO 格式标注 txt
        val/    <-- YOLO 格式标注 txt
      data.yaml
"""
from ultralytics import YOLO
import os

def train_yolo():
    # 加载现有模型作为预训练权重
    model = YOLO('cfg/best.pt')
    
    # 训练配置
    model.train(
        data='datasets/vehicle_brand/data.yaml',  # 数据集配置文件路径
        epochs=100,           # 训练轮数
        imgsz=640,            # 输入尺寸
        batch=8,              # 批次大小（根据显存调整）
        workers=4,            # 数据加载线程
        patience=20,          # 早停耐心值
        device=0,             # GPU 设备号，cpu 则填 'cpu'
        project='runs/train', # 训练结果保存路径
        name='vehicle_brand_v2',  # 本次训练名称
        exist_ok=True,        # 覆盖已有结果
        pretrained=True,      # 使用预训练权重
        optimizer='AdamW',    # 优化器
        lr0=0.001,            # 初始学习率
        lrf=0.01,             # 最终学习率 = lr0 * lrf
        augment=True,         # 数据增强
        mosaic=1.0,           # Mosaic 增强概率
        mixup=0.1,            # MixUp 增强概率
        hsv_h=0.015,          # HSV 色调增强
        hsv_s=0.7,            # HSV 饱和度增强
        hsv_v=0.4,            # HSV 亮度增强
        degrees=5.0,          # 旋转角度
        translate=0.1,        # 平移
        scale=0.5,            # 缩放
        shear=2.0,            # 剪切
        perspective=0.0,      # 透视变换
        flipud=0.0,           # 上下翻转
        fliplr=0.5,           # 左右翻转
        copy_paste=0.1,       # Copy-Paste 增强
    )
    
    # 验证
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    # 导出为 ONNX（可选）
    # model.export(format='onnx', imgsz=640)

if __name__ == '__main__':
    train_yolo()
