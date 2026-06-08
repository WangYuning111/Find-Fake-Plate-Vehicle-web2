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

csv_file = 'brand_labels.csv'
with open(csv_file, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

changed = 0
for row in rows:
    if row['brand'] == '其他':
        pred, conf = predict(row['image_path'])
        if pred and conf >= 0.90:
            row['brand'] = pred
            changed += 1

with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'批量自动标注完成，共修正 {changed} 张图片（置信度>=0.90）')
