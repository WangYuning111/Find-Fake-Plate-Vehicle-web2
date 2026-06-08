import random
import csv, cv2, torch, torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter

DEVICE = torch.device('cpu')

# 动态获取品牌列表
BRANDS = sorted(set(r['brand'] for r in csv.DictReader(open('brand_labels.csv',encoding='utf-8'))))
print("品牌列表:", BRANDS)

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(BRANDS))
model.load_state_dict(torch.load('cfg/vehicle_brand_resnet18.pth', map_location=DEVICE))
model.to(DEVICE); model.eval()

def predict(path):
    img = cv2.imread(path)
    if img is None: return '未知', 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        return BRANDS[pred.item()], conf.item()

rows = list(csv.DictReader(open('brand_labels.csv', encoding='utf-8')))
brand_counts = Counter(r['brand'] for r in rows if r['brand'] != '其他')
valid_brands = {b for b, c in brand_counts.items() if c >= 5}
valid_rows = [r for r in rows if r['brand'] in valid_brands]

total_correct = 0
for trial in range(20):
    random.seed(300 + trial)
    sample = random.sample(valid_rows, 10)
    correct = 0
    for r in sample:
        pred, conf = predict(r['image_path'])
        ok = pred == r['brand']
        if ok: 
            correct += 1
        else:
            print('  错误: {} 真实={} 预测={} 置信度={:.2f}'.format(r['image_path'], r['brand'], pred, conf))
    total_correct += correct
    print('第{}轮: {}/10 = {}%'.format(trial+1, correct, correct*10))

print('\n20轮平均正确率: {}/200 = {}%'.format(total_correct, total_correct/2))
