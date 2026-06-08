import csv
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device('cpu')
BRANDS = ['一汽','东风','丰田','五菱','别克','大众','奇瑞','奥迪','日产','本田','标致','比亚迪','江淮','江铃','海马','现代','福特','福田','舒驰','起亚','金杯','铃木','长城','长安','雪佛兰','雪铁龙']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(BRANDS))
model.load_state_dict(torch.load('cfg/vehicle_brand_resnet18_filtered.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(path):
    img = cv2.imread(path)
    if img is None:
        return None, 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        return BRANDS[pred.item()], conf.item()

# 读取所有其他标签的图片
rows = list(csv.DictReader(open('brand_labels.csv', encoding='utf-8')))
other_rows = [r for r in rows if r['brand'] == '其他']

print(f'其他标签图片总数: {len(other_rows)}')
print('\n高置信度预测结果（>=0.80）:')
print('-' * 60)

candidates = []
for row in other_rows:
    path = row['image_path']
    pred, conf = predict(path)
    if pred and conf >= 0.80:
        candidates.append((path, pred, conf))
        print(f'{path} -> {pred} (置信度: {conf:.2f})')

print(f'\n共找到 {len(candidates)} 个高置信度候选')
