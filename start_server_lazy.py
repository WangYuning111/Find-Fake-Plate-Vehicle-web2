from flask import Flask, render_template, redirect, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
import pandas as pd
import torch
import torch.nn.functional as F

# 延迟加载的变量
yolov12_model = None
v_type_model = None
v_color_model = None
aap_type = None
aap_color = None
iap = None
plateRecog = None

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
cwd_path = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    global yolov12_model, v_type_model, v_color_model, aap_type, aap_color, iap, plateRecog
    if yolov12_model is None:
        from ultralytics import YOLO
        from conv import MiniVGGNet
        from preprocessing import AspectAwarePreprocessor, ImageToTensorPreprocessor
        from hyperlpr import HyperLPR_plate_recognition as plate_recog
        
        print("Loading models...")
        yolov12_model = YOLO('cfg/best.pt')
        print(f"[MODEL] YOLO classes: {yolov12_model.names}")
        
        v_type_model = MiniVGGNet(100, 100, 3, 4).to(device)
        v_type_model.load_state_dict(torch.load("cfg/vehicle_type.pth", map_location=device))
        v_type_model.eval()
        
        v_color_model = MiniVGGNet(100, 100, 3, 8).to(device)
        v_color_model.load_state_dict(torch.load("cfg/vehicle_color.pth", map_location=device))
        v_color_model.eval()
        
        aap_type = AspectAwarePreprocessor(100, 100)
        aap_color = AspectAwarePreprocessor(100, 100)
        iap = ImageToTensorPreprocessor(data_format='channels_first')
        plateRecog = plate_recog
        print("Models loaded!")

v_type_classes = ["bus", "car", "minibus", "truck"]
v_color_classes = ["black", "blue", "brown", "green", "red", "silver", "white", "yellow"]

# 品牌名称修正映射表（当模型训练标签有误时使用）
BRAND_CORRECTION_MAP = {
    # 模型将大众车标识别为"别克"，修正为"大众"
    "别克": "大众",
}

def correct_brand(raw_brand):
    """修正品牌名称"""
    return BRAND_CORRECTION_MAP.get(raw_brand, raw_brand)

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

def predict(imgPath, vehicle_info_database):
    load_models()  # 确保模型已加载
    
    plateNo = "未知"
    carBrandZh = "未知"
    predictResult = "未识别"

    original_image = cv2.imread(imgPath)
    if original_image is None:
        app.logger.error(f"[ERROR] Cannot read image: {imgPath}")
        return plateNo, "未知", "未知", carBrandZh, "图片读取失败"
    
    app.logger.info(f"[INFO] Image size: {original_image.shape}")
    
    # 使用较低置信度阈值检测，确保能检测到车辆
    results = yolov12_model(source=original_image, conf=0.1, iou=0.45)
    result = results[0]

    app.logger.info(f"[INFO] Detection boxes: {len(result.boxes)}")
    
    # 打印模型类别信息（首次运行时）
    if result.names:
        app.logger.info(f"[INFO] Model classes: {result.names}")

    if len(result.boxes) == 0:
        app.logger.warning("[WARN] No targets detected, trying lower threshold...")
        results = yolov12_model(source=original_image, conf=0.05, iou=0.3)
        result = results[0]
        app.logger.info(f"[INFO] Second detection boxes: {len(result.boxes)}")

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    if len(boxes) == 0:
        return plateNo, "未知", "未知", carBrandZh, "未检测到车辆"

    # 选择置信度最高的检测框
    best_idx = int(confs.argmax())
    app.logger.info(f"[INFO] Best box index: {best_idx}, conf: {confs[best_idx]:.3f}")
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_id = int(classes[i])
        carBrandZh = names.get(cls_id, f"class_{cls_id}")
        confidence = float(confs[i])
        app.logger.info(f"[INFO] Box{i}: {carBrandZh}, conf: {confidence:.3f}, coords: ({x1},{y1},{x2},{y2})")

    # 使用最佳检测框
    i = best_idx
    x1, y1, x2, y2 = boxes[i]
    cls_id = int(classes[i])
    raw_brand = names.get(cls_id, f"class_{cls_id}")
    carBrandZh = correct_brand(raw_brand)
    app.logger.info(f"[INFO] Raw brand: {raw_brand}, corrected: {carBrandZh}")
    
    # 确保坐标有效
    h, w = original_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return plateNo, "未知", "未知", carBrandZh, "检测框无效"

    crop_image = original_image[y1:y2, x1:x2]
    if crop_image.size == 0:
        return plateNo, "未知", "未知", carBrandZh, "裁剪区域无效"

    type_roi = aap_type.preprocess(crop_image)
    type_tensor = iap.preprocess(type_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        type_output = v_type_model(type_tensor)
        type_probs = torch.softmax(type_output, dim=1)[0].cpu().numpy()
    v_type_result = dict(zip(v_type_classes, type_probs))
    v_type_label = max(v_type_result, key=v_type_result.get)
    app.logger.info(f"[INFO] Type prediction: {v_type_label}")

    color_roi = aap_color.preprocess(crop_image)
    color_tensor = iap.preprocess(color_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        color_output = v_color_model(color_tensor)
        color_probs = torch.softmax(color_output, dim=1)[0].cpu().numpy()
    v_color_result = dict(zip(v_color_classes, color_probs))
    v_color_label = max(v_color_result, key=v_color_result.get)
    app.logger.info(f"[INFO] Color prediction: {v_color_label}")

    # 识别车牌 - 尝试多种策略，优先使用高分辨率图像
    plateInfo = None
    
    # 策略1: 对原始全图进行车牌识别（分辨率最高，效果最好）
    plateInfo = plateRecog(original_image)
    app.logger.info(f"[INFO] Plate recognition (full): {plateInfo}")
    
    # 策略2: 对全图放大后识别
    if not plateInfo or len(plateInfo) == 0:
        h_o, w_o = original_image.shape[:2]
        if h_o < 1000 or w_o < 1000:
            scale = min(2.0, 2000 / max(h_o, w_o))
            enlarged_full = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            plateInfo = plateRecog(enlarged_full)
            app.logger.info(f"[INFO] Plate recognition (full scaled {scale:.1f}x): {plateInfo}")
    
    # 策略3: 对车辆裁剪图进行车牌识别
    if not plateInfo or len(plateInfo) == 0:
        if crop_image.size > 0:
            plateInfo = plateRecog(crop_image)
            app.logger.info(f"[INFO] Plate recognition (crop): {plateInfo}")
    
    # 策略4: 只取检测框下半部分（车牌通常在下方）放大识别
    if not plateInfo or len(plateInfo) == 0:
        h_c, w_c = crop_image.shape[:2]
        bottom_half = crop_image[int(h_c*0.55):int(h_c*0.95), int(w_c*0.1):int(w_c*0.9)]
        if bottom_half.size > 0:
            bottom_half = cv2.resize(bottom_half, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            plateInfo = plateRecog(bottom_half)
            app.logger.info(f"[INFO] Plate recognition (bottom half scaled): {plateInfo}")
    
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
            app.logger.info(f"[INFO] Final plate: {plateNo}, conf: {conf:.3f}")
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

@app.before_request
def before_request():
    if not hasattr(app, 'vehicle_info_database'):
        app.vehicle_info_database = pd.read_csv('vehicle-database.csv')

@app.route('/prepare')
def warm_up():
    app.vehicle_info_database = pd.read_csv('vehicle-database.csv')
    return redirect('/')

@app.route('/', methods=['POST', 'GET'])
def analyze():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename == '':
            return render_template('index.html', error='请选择要上传的图片')
        
        if not allowed_file(f.filename):
            return render_template('index.html', error='只支持 png、jpg、bmp 格式')

        try:
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
            f.save(upload_path)

            test_img_path = os.path.join(basepath, 'static/images/test.jpg')
            img = cv2.imread(upload_path)
            if img is None:
                return render_template('index.html', error='无法读取图片，请检查图片格式')
            cv2.imwrite(test_img_path, img)

            plate_no, v_type, v_color, car_brand, predict_result = predict(test_img_path, app.vehicle_info_database)
            context = [
                f"车牌号: {plate_no}",
                f"车辆类型: {v_type}",
                f"车辆颜色: {v_color}",
                f"车辆品牌: {car_brand}",
                f"判定结果: {predict_result}"
            ]
            return render_template('index.html', context=context, val1=time.time())
        except Exception as e:
            app.logger.error(f'处理图片时出错: {str(e)}')
            return render_template('index.html', error=f'处理失败: {str(e)}')

    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API接口：接收图片并返回JSON格式结果"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "没有上传文件"}), 400
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({"success": False, "error": "文件名为空"}), 400
    
    if not allowed_file(f.filename):
        return jsonify({"success": False, "error": "不支持的文件格式"}), 400
    
    try:
        load_models()
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
        
        test_img_path = os.path.join(basepath, 'static/images/test.jpg')
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({"success": False, "error": "无法读取图片"}), 400
        cv2.imwrite(test_img_path, img)
        
        if not hasattr(app, 'vehicle_info_database'):
            app.vehicle_info_database = pd.read_csv('vehicle-database.csv')
        
        plate_no, v_type, v_color, car_brand, predict_result = predict(test_img_path, app.vehicle_info_database)
        
        return jsonify({
            "success": True,
            "data": {
                "plate_number": plate_no,
                "vehicle_type": v_type,
                "vehicle_color": v_color,
                "vehicle_brand": car_brand,
                "result": predict_result,
                "is_fake": predict_result == "套牌车",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "service": "套牌车智能稽查系统",
        "version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    print("=" * 50)
    print("套牌车智能稽查系统 V1.0")
    print("=" * 50)
    print("访问地址: http://127.0.0.1:8090")
    print("API文档: http://127.0.0.1:8090/health")
    print("=" * 50)
    app.run(port=8090, debug=True, use_reloader=False, threaded=True)
