#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机抽取10张已标注的非'其他'图片测试品牌识别正确率
"""
import random
random.seed(42)
import csv
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter

DEVICE = torch.device('cpu')
BRANDS = ['大众','别克','福特','东风','长安','日产','现代','一汽','江淮','金杯',
          '福田','江铃','哈飞','雪铁龙','中华','雪佛兰','奥迪','长城','标致',
          '马自达','铃木','五菱','海马','宝马','荣威','比亚迪','奇瑞','本田','丰田','起亚','奔驰','舒驰','其他']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
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

rows = list(csv.DictReader(open('brand_labels.csv', encoding='utf-8')))
# 只从样本数>=5的品牌中抽取
brand_counts = Counter(r['brand'] for r in rows if r['brand'] != '其他')
valid_brands = {b for b, c in brand_counts.items() if c >= 5}
valid_rows = [r for r in rows if r['brand'] in valid_brands]

print(f"可选品牌(>=5样本): {sorted(valid_brands)}")
print(f"可选图片数: {len(valid_rows)}")

sample = random.sample(valid_rows, 10)
correct = 0
for r in sample:
    pred, conf = predict(r['image_path'])
    ok = pred == r['brand']
    if ok:
        correct += 1
    status = 'OK' if ok else 'NG'
    print(f"[{status}] {r['image_path']}: 真实={r['brand']}, 预测={pred}, 置信度={conf:.2f}")

print(f"\n正确率: {correct}/10 = {correct*10}%")
