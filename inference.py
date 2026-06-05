#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理封装
- 统一模型加载（带缓存）
- 单图推理 + 批量推理
- 检测结果可视化（画框+标签）
- 错误分类与降级处理
"""
import os
import cv2
import time
import torch
import numpy as np
from config import Config

# ============== 模型缓存 ==============
_models = {}

def get_device():
    """获取计算设备"""
    if Config.DEVICE == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(Config.DEVICE)


def load_models(force_reload=False):
    """加载所有模型（线程安全，只加载一次）"""
    global _models
    if _models and not force_reload:
        return _models

    device = get_device()
    print(f"[INFERENCE] 加载模型到设备: {device}")

    from ultralytics import YOLO
    from conv import MiniVGGNet
    from preprocessing import AspectAwarePreprocessor, ImageToTensorPreprocessor
    from hyperlpr import HyperLPR_plate_recognition as plate_recog

    # YOLOv12 检测
    yolo = YOLO(Config.YOLO_MODEL_PATH)
    print(f"[INFERENCE] YOLO 类别: {yolo.names}")

    # 车型分类
    v_type_model = MiniVGGNet(100, 100, 3, 4).to(device)
    v_type_model.load_state_dict(torch.load(Config.VEHICLE_TYPE_MODEL_PATH, map_location=device, weights_only=True))
    v_type_model.eval()

    # 颜色分类
    v_color_model = MiniVGGNet(100, 100, 3, 8).to(device)
    v_color_model.load_state_dict(torch.load(Config.VEHICLE_COLOR_MODEL_PATH, map_location=device, weights_only=True))
    v_color_model.eval()

    # 预处理器
    aap = AspectAwarePreprocessor(100, 100)
    iap = ImageToTensorPreprocessor(data_format='channels_first')

    _models = {
        'yolo': yolo,
        'type_model': v_type_model,
        'color_model': v_color_model,
        'aap': aap,
        'iap': iap,
        'plate_recog': plate_recog,
        'device': device,
        'type_classes': ["bus", "car", "minibus", "truck"],
        'color_classes': ["black", "blue", "brown", "green", "red", "silver", "white", "yellow"],
    }
    print("[INFERENCE] 所有模型加载完成")
    return _models


# ============== 推理异常分类 ==============
class InferenceError(Exception):
    """推理基础异常"""
    pass


class ImageLoadError(InferenceError):
    """图片读取失败"""
    pass


class NoVehicleError(InferenceError):
    """未检测到车辆"""
    pass


class PlateRecognizeError(InferenceError):
    """车牌识别失败"""
    pass


# ============== 核心推理 ==============

def preprocess_image(img_path):
    """读取并预处理图片"""
    img = cv2.imread(img_path)
    if img is None:
        raise ImageLoadError(f"无法读取图片: {img_path}")
    return img


def detect_vehicle(img, models):
    """YOLO 车辆检测，返回所有检测结果"""
    yolo = models['yolo']
    device = models['device']

    # 第一次检测
    results = yolo(source=img, conf=Config.YOLO_CONF, iou=Config.YOLO_IOU, verbose=False)
    result = results[0]

    # 如果没检测到，降低阈值再试一次
    if len(result.boxes) == 0:
        results = yolo(source=img, conf=0.05, iou=0.3, verbose=False)
        result = results[0]

    if len(result.boxes) == 0:
        raise NoVehicleError("未在图片中检测到车辆")

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_id = int(classes[i])
        detections.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'class_id': cls_id,
            'class_name': names.get(cls_id, f"class_{cls_id}"),
            'confidence': float(confs[i]),
        })

    return detections


def classify_crop(crop, models):
    """对车辆裁剪图进行车型和颜色分类"""
    aap = models['aap']
    iap = models['iap']
    device = models['device']

    # 车型
    type_roi = aap.preprocess(crop)
    type_tensor = iap.preprocess(type_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        type_output = models['type_model'](type_tensor)
        type_probs = torch.softmax(type_output, dim=1)[0].cpu().numpy()
    v_type = models['type_classes'][int(np.argmax(type_probs))]
    type_conf = float(np.max(type_probs))

    # 颜色
    color_roi = aap.preprocess(crop)
    color_tensor = iap.preprocess(color_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        color_output = models['color_model'](color_tensor)
        color_probs = torch.softmax(color_output, dim=1)[0].cpu().numpy()
    v_color = models['color_classes'][int(np.argmax(color_probs))]
    color_conf = float(np.max(color_probs))

    return v_type, type_conf, v_color, color_conf


def recognize_plate(img, crop=None):
    """多级车牌识别策略"""
    plateRecog = _models['plate_recog']

    strategies = [
        ("full", lambda: plateRecog(img)),
    ]

    # 策略2: 全图放大
    h_o, w_o = img.shape[:2]
    if h_o < 1000 or w_o < 1000:
        scale = min(2.0, 2000 / max(h_o, w_o))
        enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        strategies.append(("scaled", lambda: plateRecog(enlarged)))

    # 策略3: 裁剪图
    if crop is not None and crop.size > 0:
        strategies.append(("crop", lambda: plateRecog(crop)))
        # 策略4: 裁剪图下半部分
        h_c, w_c = crop.shape[:2]
        bottom = crop[int(h_c*0.55):int(h_c*0.95), int(w_c*0.1):int(w_c*0.9)]
        if bottom.size > 0:
            bottom = cv2.resize(bottom, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            strategies.append(("bottom", lambda: plateRecog(bottom)))

    all_results = []
    for name, fn in strategies:
        try:
            result = fn()
            if result and len(result) > 0:
                all_results.extend(result)
        except Exception as e:
            continue

    if not all_results:
        raise PlateRecognizeError("所有车牌识别策略均失败")

    # 过滤和排序
    valid = []
    for p in all_results:
        if len(p) > 1:
            text = str(p[0]).replace('·', '').replace('-', '')
            conf = float(p[1]) if len(p) > 1 else 0
            if len(text) >= Config.PLATE_MIN_LENGTH:
                valid.append((str(p[0]), conf))

    if not valid:
        raise PlateRecognizeError("未能通过置信度过滤")

    best = max(valid, key=lambda x: x[1])
    return best[0], best[1]


def draw_result(img, detections, selected_idx, plate, v_type, v_color, brand, is_fake):
    """在图片上绘制检测框和结果标签"""
    vis = img.copy()
    h, w = vis.shape[:2]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0) if i == selected_idx else (128, 128, 128)
        thickness = 3 if i == selected_idx else 1

        # 画框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # 标签
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 底部信息栏
    info_lines = [
        f"Plate: {plate}",
        f"Type: {v_type} | Color: {v_color}",
        f"Brand: {brand}",
        f"Result: {'FAKE' if is_fake else 'NORMAL'}"
    ]
    bar_h = 30 * len(info_lines) + 20
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    for i, line in enumerate(info_lines):
        y = h - bar_h + 30 + i * 28
        color = (0, 0, 255) if is_fake and i == len(info_lines) - 1 else (0, 255, 0)
        cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return vis


def predict_single(img_path, save_result_path=None):
    """
    单张图片完整推理流程
    返回: {
        'plate_number': str,
        'vehicle_type': str,
        'vehicle_color': str,
        'vehicle_brand': str,
        'is_fake': bool,
        'true_brand': str,
        'confidence': float,
        'result_image': str or None,
        'error': str or None,
    }
    """
    models = load_models()
    start_time = time.time()

    result = {
        'plate_number': '未知',
        'vehicle_type': '未知',
        'vehicle_color': '未知',
        'vehicle_brand': '未知',
        'is_fake': False,
        'true_brand': None,
        'confidence': 0.0,
        'result_image': None,
        'error': None,
        'process_time': 0.0,
    }

    try:
        # 1. 读取图片
        img = preprocess_image(img_path)

        # 2. 车辆检测
        detections = detect_vehicle(img, models)
        best_det = max(detections, key=lambda x: x['confidence'])

        # 3. 裁剪车辆区域
        x1, y1, x2, y2 = best_det['bbox']
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            raise InferenceError("车辆裁剪区域无效")

        # 4. 车型/颜色分类
        v_type, type_conf, v_color, color_conf = classify_crop(crop, models)

        # 5. 品牌（从 YOLO 检测类别）
        brand = best_det['class_name']
        brand_conf = best_det['confidence']

        # 6. 车牌识别
        try:
            plate, plate_conf = recognize_plate(img, crop)
            result['plate_number'] = plate
        except PlateRecognizeError:
            plate = '未知'
            plate_conf = 0.0
            result['plate_number'] = '未知'

        # 7. 套牌检测
        from database import check_fake_plate
        is_fake, true_brand, _ = check_fake_plate(plate, brand)

        # 8. 综合置信度
        overall_conf = (brand_conf + type_conf + color_conf + plate_conf) / 4

        # 9. 可视化
        if save_result_path:
            vis_img = draw_result(img, detections, detections.index(best_det),
                                  plate, v_type, v_color, brand, is_fake)
            cv2.imwrite(save_result_path, vis_img)
            result['result_image'] = save_result_path

        result.update({
            'vehicle_type': v_type,
            'vehicle_color': v_color,
            'vehicle_brand': brand,
            'is_fake': is_fake,
            'true_brand': true_brand,
            'confidence': round(overall_conf, 4),
            'process_time': round(time.time() - start_time, 3),
        })

    except ImageLoadError as e:
        result['error'] = f"图片读取失败: {str(e)}"
    except NoVehicleError as e:
        result['error'] = str(e)
    except InferenceError as e:
        result['error'] = f"推理错误: {str(e)}"
    except Exception as e:
        result['error'] = f"未知错误: {str(e)}"
        import traceback
        traceback.print_exc()

    return result
