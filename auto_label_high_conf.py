#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用当前模型对'其他'标签图片进行高置信度自动标注
"""
import csv
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

DEVICE = torch.device('cpu')
BRANDS = ['大众','别克','福特','东风','长安','日产','现代','一汽','江淮','金杯',
          '福田','江铃','哈飞','雪铁龙','中华','雪佛兰','奥迪','长城','标致',
          '马自达','铃木','五菱','海马','宝马','荣威','比亚迪','奇瑞','本田','丰田','起亚','奔驰','其他']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(BRANDS))
model.load_state_dict(torch.load('cfg/vehicle_brand_resnet18.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(path):
    img = cv2.imread(path)
    if img is None:
        return '未知', 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        return BRANDS[pred.item()], conf.item()

# 只给样本数>=5的品牌自动标注，避免污染少样本品牌
brand_counts = {}
rows = list(csv.DictReader(open('brand_labels.csv', encoding='utf-8')))
for r in rows:
    b = r['brand']
    brand_counts[b] = brand_counts.get(b, 0) + 1

safe_brands = {b for b, c in brand_counts.items() if c >= 5}
print(f"安全品牌(>=5样本): {safe_brands}")

auto_labeled = 0
for r in rows:
    if r['brand'] == '其他':
        pred, conf = predict(r['image_path'])
        if pred in safe_brands and conf >= 0.50 and pred != '其他':
            r['brand'] = pred
            auto_labeled += 1
            if auto_labeled <= 20:
                print(f"自动标注: {r['image_path']} -> {pred} (置信度{conf:.2f})")

with open('brand_labels.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_path', 'brand', 'timestamp'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n自动标注完成: {auto_labeled} 张")
