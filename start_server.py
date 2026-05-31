from flask import Flask, render_template, redirect, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from conv import MiniVGGNet
from preprocessing import AspectAwarePreprocessor, ImageToTensorPreprocessor, SimplePreprocessor
from hyperlpr import HyperLPR_plate_recognition as plateRecog


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
cwd_path = os.getcwd()
yolov12_model_path = 'cfg/best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolov12_model = YOLO(yolov12_model_path)  # 加载YOLOv12模型
print(f"[MODEL] YOLO classes: {yolov12_model.names}")

# 加载车辆类型分类模型
v_type_model = MiniVGGNet(100, 100, 3, 4).to(device)
v_type_model.load_state_dict(torch.load("cfg/vehicle_type.pth", map_location=device))
v_type_model.eval()
v_type_classes = ["bus", "car", "minibus", "truck"]

# 加载车辆颜色分类模型
v_color_model = MiniVGGNet(100, 100, 3, 8).to(device)
v_color_model.load_state_dict(torch.load("cfg/vehicle_color.pth", map_location=device))
v_color_model.eval()
v_color_classes = ["black", "blue", "brown", "green", "red", "silver", "white", "yellow"]

# 品牌名称修正映射表（当模型训练标签有误时使用）
BRAND_CORRECTION_MAP = {
    # 模型将大众车标识别为"别克"，修正为"大众"
    "别克": "大众",
}

def correct_brand(raw_brand):
    """修正品牌名称"""
    return BRAND_CORRECTION_MAP.get(raw_brand, raw_brand)

# 初始化图像预处理器
aap_type = AspectAwarePreprocessor(100, 100)
aap_color = AspectAwarePreprocessor(100, 100)
iap = ImageToTensorPreprocessor(data_format='channels_first')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def isFakePlate(inputCarInfo, carInfoDatabase):
    plateNo, carBrand = inputCarInfo
    if carBrand == '未知':
        return False, 'Null'
    result = carInfoDatabase[(carInfoDatabase['plateNo'] == plateNo)]
    if len(result) > 0:
        trueBrand = result['carBrand'].values[0]
        isFake = (carBrand != trueBrand)
        return isFake, trueBrand
    else:
        return False, 'Null'


def prepare_service():
    return pd.read_csv('vehicle-database.csv')


def predict(imgPath, vehicle_info_database):
    plateNo = "未知"
    carBrandZh = "未知"
    predictResult = "未识别"

    original_image = cv2.imread(imgPath)
    if original_image is None:
        print(f"[ERROR] Cannot read image: {imgPath}")
        return plateNo, "未知", "未知", carBrandZh, "图片读取失败"
    
    print(f"[INFO] Image size: {original_image.shape}")
    
    # 使用较低置信度阈值检测，确保能检测到车辆
    results = yolov12_model(source=original_image, conf=0.1, iou=0.45)
    result = results[0]

    print(f"[INFO] Detection boxes: {len(result.boxes)}")
    if result.names:
        print(f"[INFO] Model classes: {result.names}")

    if len(result.boxes) == 0:
        print("[WARN] No targets detected, trying lower threshold...")
        results = yolov12_model(source=original_image, conf=0.05, iou=0.3)
        result = results[0]
        print(f"[INFO] Second detection boxes: {len(result.boxes)}")

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    if len(boxes) == 0:
        return plateNo, "未知", "未知", carBrandZh, "未检测到车辆"

    # 选择置信度最高的检测框
    best_idx = int(confs.argmax())
    print(f"[INFO] Best box index: {best_idx}, conf: {confs[best_idx]:.3f}")
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_id = int(classes[i])
        print(f"[INFO] Box{i}: {names.get(cls_id, f'class_{cls_id}')}, conf: {confs[i]:.3f}")

    i = best_idx
    x1, y1, x2, y2 = boxes[i]
    cls_id = int(classes[i])
    raw_brand = names.get(cls_id, f"class_{cls_id}")
    carBrandZh = correct_brand(raw_brand)
    print(f"[INFO] Raw brand: {raw_brand}, corrected: {carBrandZh}")
    
    # 确保坐标有效
    h, w = original_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return plateNo, "未知", "未知", carBrandZh, "检测框无效"

    crop_image = original_image[y1:y2, x1:x2]
    if crop_image.size == 0:
        return plateNo, "未知", "未知", carBrandZh, "裁剪区域无效"

    # 使用车型模型进行预测
    type_roi = aap_type.preprocess(crop_image)
    type_tensor = iap.preprocess(type_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        type_output = v_type_model(type_tensor)
        type_probs = torch.softmax(type_output, dim=1)[0].cpu().numpy()
    v_type_result = dict(zip(v_type_classes, type_probs))
    v_type_label = max(v_type_result, key=v_type_result.get)
    print(f"[INFO] Type prediction: {v_type_label}")

    # 使用车辆颜色模型进行预测
    color_roi = aap_color.preprocess(crop_image)
    color_tensor = iap.preprocess(color_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        color_output = v_color_model(color_tensor)
        color_probs = torch.softmax(color_output, dim=1)[0].cpu().numpy()
    v_color_result = dict(zip(v_color_classes, color_probs))
    v_color_label = max(v_color_result, key=v_color_result.get)
    print(f"[INFO] Color prediction: {v_color_label}")

    # 识别车牌 - 尝试多种策略，优先使用高分辨率图像
    plateInfo = None
    
    # 策略1: 对原始全图进行车牌识别（分辨率最高，效果最好）
    plateInfo = plateRecog(original_image)
    print(f"[INFO] Plate recognition (full): {plateInfo}")
    
    # 策略2: 对全图放大后识别
    if not plateInfo or len(plateInfo) == 0:
        h_o, w_o = original_image.shape[:2]
        if h_o < 1000 or w_o < 1000:
            scale = min(2.0, 2000 / max(h_o, w_o))
            enlarged_full = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            plateInfo = plateRecog(enlarged_full)
            print(f"[INFO] Plate recognition (full scaled {scale:.1f}x): {plateInfo}")
    
    # 策略3: 对车辆裁剪图进行车牌识别
    if not plateInfo or len(plateInfo) == 0:
        if crop_image.size > 0:
            plateInfo = plateRecog(crop_image)
            print(f"[INFO] Plate recognition (crop): {plateInfo}")
    
    # 策略4: 只取检测框下半部分（车牌通常在下方）放大识别
    if not plateInfo or len(plateInfo) == 0:
        h_c, w_c = crop_image.shape[:2]
        bottom_half = crop_image[int(h_c*0.55):int(h_c*0.95), int(w_c*0.1):int(w_c*0.9)]
        if bottom_half.size > 0:
            bottom_half = cv2.resize(bottom_half, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            plateInfo = plateRecog(bottom_half)
            print(f"[INFO] Plate recognition (bottom half scaled): {plateInfo}")
    
    # 置信度阈值
    CONFIDENCE_THRESHOLD = 0.75

    if plateInfo and len(plateInfo) > 0:
        # 过滤：长度>=5 且 置信度>=阈值
        valid_plates = [
            p for p in plateInfo
            if len(p) > 1 and len(str(p[0]).replace('·', '').replace('-', '')) >= 5 and float(p[1]) >= CONFIDENCE_THRESHOLD
        ]
        
        # 如果严格过滤后为空，放宽条件再选一次
        if not valid_plates:
            valid_plates = [
                p for p in plateInfo
                if len(p) > 1 and len(str(p[0]).replace('·', '').replace('-', '')) >= 5
            ]
        
        if valid_plates:
            best_plate = max(valid_plates, key=lambda x: x[1] if len(x) > 1 else 0)
            plateNo = best_plate[0]
            conf = float(best_plate[1]) if len(best_plate) > 1 else 0
            print(f"[INFO] Final plate: {plateNo}, conf: {conf:.3f}")
            # 低置信度警告
            if conf < 0.85:
                plateNo = f"{plateNo} (conf {conf:.0%})"
            inputCarInfo = [plateNo, carBrandZh]
            isFake, true_car_brand = isFakePlate(inputCarInfo, vehicle_info_database)
            predictResult = "套牌车" if isFake else "正常"
        else:
            plateNo = "未知"
            predictResult = "车牌无法识别"
    else:
        plateNo = "未知"
        predictResult = "车牌无法识别"

    return plateNo, v_type_label, v_color_label, carBrandZh, predictResult


app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
vehicle_info_database = prepare_service()


@app.route('/prepare')
def warm_up():
    global vehicle_info_database
    vehicle_info_database = prepare_service()
    return redirect('/')


@app.route('/', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "只支持 png、jpg、bmp 格式"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)

        test_img_path = os.path.join(basepath, 'static/images/test.jpg')
        img = cv2.imread(upload_path)
        cv2.imwrite(test_img_path, img)

        plate_no, v_type, v_color, car_brand, predict_result = predict(test_img_path, vehicle_info_database)
        context = [
            f"车牌号: {plate_no}",
            f"车辆类型: {v_type}",
            f"车辆颜色: {v_color}",
            f"车辆品牌: {car_brand}",
            f"判定结果: {predict_result}"
        ]
        return render_template('index.html', context=context, val1=time.time())

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=8090, debug=True)

