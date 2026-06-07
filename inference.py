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
from brand_corrector import correct_brand, validate_brand_type_consistency
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision import models

# 自动查找系统中文字体
def _find_chinese_font():
    """查找系统中文字体，优先微软雅黑/宋体/黑体"""
    if os.name == 'nt':  # Windows
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/PingFang.ttc",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

_FONT_PATH = _find_chinese_font()

# ============== 模型缓存 ==============
_models = {}

def get_device():
    """获取计算设备"""
    if Config.DEVICE == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(Config.DEVICE)


def _check_model_files():
    """检查模型文件是否存在，缺失时给出清晰提示"""
    missing = []
    for name, path in [
        ("YOLO 检测模型", Config.YOLO_MODEL_PATH),
        ("车型分类模型", Config.VEHICLE_TYPE_MODEL_PATH),
        ("颜色分类模型", Config.VEHICLE_COLOR_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            missing.append((name, path))
    if missing:
        msg_lines = ["[ERROR] 以下模型文件缺失，请先下载模型权重:"]
        for name, path in missing:
            msg_lines.append(f"  - {name}: {path}")
        msg_lines.append("")
        msg_lines.append("下载方式:")
        msg_lines.append("  1) 运行脚本: bash download_model.sh")
        msg_lines.append("  2) 手动下载: 见 README.md -> 模型下载")
        msg_lines.append("  3) 从原项目 cfg/ 目录复制到 weights/")
        raise FileNotFoundError("\n".join(msg_lines))


def load_models(force_reload=False):
    """加载所有模型（线程安全，只加载一次）"""
    global _models
    if _models and not force_reload:
        return _models

    # 先检查模型文件是否存在
    _check_model_files()

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

    # 品牌分类器 (ResNet18)
    brand_model = None
    brand_classes = None
    brand_transform = None
    brand_model_path = 'cfg/vehicle_brand_resnet18.pth'
    if os.path.exists(brand_model_path):
        # 先从checkpoint读取类别信息
        checkpoint = torch.load(brand_model_path, map_location=device, weights_only=True)
        # 确定类别数
        if 'fc.weight' in checkpoint:
            num_brands = checkpoint['fc.weight'].shape[0]
        else:
            num_brands = 32
        brand_model = models.resnet18(weights=None)
        brand_model.fc = torch.nn.Linear(brand_model.fc.in_features, num_brands)
        brand_model.load_state_dict(checkpoint)
        brand_model = brand_model.to(device)
        brand_model.eval()
        # 从brand_labels.csv获取类别顺序
        import csv
        brand_labels_set = set()
        if os.path.exists('brand_labels.csv'):
            with open('brand_labels.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    brand_labels_set.add(row['brand'].strip())
        brand_classes = sorted(list(brand_labels_set)) if brand_labels_set else [
            "奥迪", "宝马", "奔驰", "比亚迪", "标致", "别克", "长安", "长城", "大众", "东风",
            "福特", "福田", "哈飞", "海马", "江淮", "江铃", "金杯", "铃木", "马自达", "日产",
            "荣威", "奇瑞", "起亚", "其他", "三菱", "上汽", "五菱", "现代", "雪佛兰", "雪铁龙",
            "一汽", "中华"
        ]
        # 确保类别数匹配
        if len(brand_classes) != num_brands:
            print(f"[WARN] 品牌类别数不匹配: csv={len(brand_classes)}, model={num_brands}")
            # 用默认值补足或截断
            default_classes = [
                "奥迪", "宝马", "奔驰", "比亚迪", "标致", "别克", "长安", "长城", "大众", "东风",
                "福特", "福田", "哈飞", "海马", "江淮", "江铃", "金杯", "铃木", "马自达", "日产",
                "荣威", "奇瑞", "起亚", "其他", "三菱", "上汽", "五菱", "现代", "雪佛兰", "雪铁龙",
                "一汽", "中华"
            ]
            if len(brand_classes) < num_brands:
                for c in default_classes:
                    if c not in brand_classes:
                        brand_classes.append(c)
                    if len(brand_classes) == num_brands:
                        break
            brand_classes = brand_classes[:num_brands]
        brand_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[INFERENCE] 品牌分类器加载完成: {num_brands} 个品牌")
    else:
        print("[INFERENCE] 品牌分类器未找到，使用YOLO品牌输出")

    # 预处理器
    aap = AspectAwarePreprocessor(100, 100)
    iap = ImageToTensorPreprocessor(data_format='channels_first')

    _models = {
        'yolo': yolo,
        'type_model': v_type_model,
        'color_model': v_color_model,
        'brand_model': brand_model,
        'brand_classes': brand_classes,
        'brand_transform': brand_transform,
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
    """YOLO 车辆检测，返回所有检测结果（支持多尺度检测）"""
    yolo = models['yolo']
    device = models['device']

    all_detections = []
    h, w = img.shape[:2]

    # 策略1: 原始尺寸检测
    results = yolo(source=img, conf=Config.YOLO_CONF, iou=Config.YOLO_IOU, verbose=False)
    result = results[0]
    if len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            all_detections.append({
                'bbox': tuple(result.boxes.xyxy.cpu().numpy().astype(int)[i]),
                'class_id': int(result.boxes.cls.cpu().numpy()[i]),
                'class_name': result.names.get(int(result.boxes.cls.cpu().numpy()[i]), "unknown"),
                'confidence': float(result.boxes.conf.cpu().numpy()[i]),
                'scale': 'original',
            })

    # 策略2: 如果原始尺寸没检测到或小图，尝试放大
    if len(all_detections) == 0 or max(h, w) < 800:
        scale = 2.0 if max(h, w) < 600 else 1.5
        enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results = yolo(source=enlarged, conf=0.05, iou=0.3, verbose=False)
        result = results[0]
        if len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[i]
                # 将坐标缩放回原始尺寸
                all_detections.append({
                    'bbox': (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)),
                    'class_id': int(result.boxes.cls.cpu().numpy()[i]),
                    'class_name': result.names.get(int(result.boxes.cls.cpu().numpy()[i]), "unknown"),
                    'confidence': float(result.boxes.conf.cpu().numpy()[i]) * 0.9,  # 放大检测置信度惩罚
                    'scale': f'enlarged_{scale}x',
                })

    # 策略3: 如果还是没检测到，尝试旋转图片（处理旋转数据增强的图片）
    if len(all_detections) == 0:
        for angle in [15, -15, 30, -30, 45, -45]:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
            results = yolo(source=rotated, conf=0.05, iou=0.3, verbose=False)
            result = results[0]
            if len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[i]
                    # 将坐标旋转回原始角度
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    pts_rotated = cv2.transform(np.array([pts]), M_inv)[0]
                    rx1 = int(min(pts_rotated[:, 0]))
                    ry1 = int(min(pts_rotated[:, 1]))
                    rx2 = int(max(pts_rotated[:, 0]))
                    ry2 = int(max(pts_rotated[:, 1]))
                    all_detections.append({
                        'bbox': (rx1, ry1, rx2, ry2),
                        'class_id': int(result.boxes.cls.cpu().numpy()[i]),
                        'class_name': result.names.get(int(result.boxes.cls.cpu().numpy()[i]), "unknown"),
                        'confidence': float(result.boxes.conf.cpu().numpy()[i]) * 0.7,
                        'scale': f'rotated_{angle}',
                    })
                break  # 找到一个就退出

    # 策略4: 如果还是没检测到，进一步降低阈值
    if len(all_detections) == 0:
        results = yolo(source=img, conf=0.01, iou=0.3, verbose=False)
        result = results[0]
        if len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                all_detections.append({
                    'bbox': tuple(result.boxes.xyxy.cpu().numpy().astype(int)[i]),
                    'class_id': int(result.boxes.cls.cpu().numpy()[i]),
                    'class_name': result.names.get(int(result.boxes.cls.cpu().numpy()[i]), "unknown"),
                    'confidence': float(result.boxes.conf.cpu().numpy()[i]) * 0.8,
                    'scale': 'low_conf',
                })

    if len(all_detections) == 0:
        raise NoVehicleError("未在图片中检测到车辆")

    # 去重：如果同一个目标被多次检测，保留置信度最高的
    # 使用IOU去重
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        return inter / (area1 + area2 - inter + 1e-6)

    # 按置信度排序，保留不重叠的检测
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    final_detections = []
    for det in all_detections:
        is_duplicate = False
        for existing in final_detections:
            if iou(det['bbox'], existing['bbox']) > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append(det)

    return final_detections


def classify_crop(crop, models, full_img=None, bbox=None):
    """对车辆进行车型和颜色分类"""
    aap = models['aap']
    iap = models['iap']
    device = models['device']

    # 使用全图进行分类（与训练时输入一致）
    classify_img = full_img if full_img is not None else crop

    # 车型 (MiniVGGNet，输入全图)
    type_roi = aap.preprocess(classify_img)
    type_tensor = iap.preprocess(type_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        type_output = models['type_model'](type_tensor)
        type_probs = torch.softmax(type_output, dim=1)[0].cpu().numpy()
    v_type = models['type_classes'][int(np.argmax(type_probs))]
    type_conf = float(np.max(type_probs))

    # 颜色 (MiniVGGNet，输入全图)
    color_roi = aap.preprocess(classify_img)
    color_tensor = iap.preprocess(color_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        color_output = models['color_model'](color_tensor)
        color_probs = torch.softmax(color_output, dim=1)[0].cpu().numpy()
    v_color = models['color_classes'][int(np.argmax(color_probs))]
    color_conf = float(np.max(color_probs))

    return v_type, type_conf, v_color, color_conf


# 中国车牌省份简称和校验规则
_PLATE_PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
_PLATE_ALPHABET = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
_PLATE_DIGITS = set("0123456789")


def _validate_plate(plate_text):
    """
    车牌号基本格式校验
    返回: (is_valid, corrected_text)
    """
    if not plate_text or len(plate_text) < 6:
        return False, plate_text

    # 清理字符
    text = plate_text.upper().replace('·', '').replace('-', '').replace(' ', '').replace('。', 'O').replace('｜', 'I')

    # 首字符必须是省份简称
    if text[0] not in _PLATE_PROVINCES:
        # 尝试修复常见OCR错误
        fix_map = {'0': '沪', '1': '京', '2': '津', '3': '冀', '4': '晋', '5': '蒙', '6': '辽',
                   '7': '吉', '8': '黑', '9': '苏'}
        if text[0] in fix_map:
            text = fix_map[text[0]] + text[1:]
        else:
            return False, plate_text

    # 第二位必须是字母
    if len(text) >= 2 and text[1] not in _PLATE_ALPHABET:
        return False, plate_text

    # 长度校验：普通车牌 7 位（如京A12345），新能源 8 位（如京AD12345）
    if len(text) == 7 or len(text) == 8:
        return True, text

    return False, plate_text


def _postprocess_plate_results(all_results):
    """
    对多策略识别结果进行投票融合 + 校验
    返回: (best_plate, confidence)
    """
    # 收集所有结果并统计出现频次
    from collections import Counter

    vote_pool = []
    for text, conf, strategy in all_results:
        text_clean = str(text).replace('·', '').replace('-', '').replace(' ', '')
        is_valid, corrected = _validate_plate(text_clean)
        if is_valid:
            # 通过校验的结果，置信度加成
            vote_pool.append((corrected, conf * 1.1, strategy))
        else:
            # 未通过校验，置信度惩罚
            vote_pool.append((text_clean, conf * 0.5, strategy))

    if not vote_pool:
        return None, 0.0

    # 按出现频次 + 平均置信度投票
    plate_scores = {}
    for text, conf, _ in vote_pool:
        if text not in plate_scores:
            plate_scores[text] = {'count': 0, 'total_conf': 0.0}
        plate_scores[text]['count'] += 1
        plate_scores[text]['total_conf'] += conf

    # 综合得分 = 出现次数 * 0.3 + 平均置信度 * 0.7
    best_plate = None
    best_score = -1
    for text, info in plate_scores.items():
        avg_conf = info['total_conf'] / info['count']
        score = info['count'] * 0.3 + avg_conf * 0.7
        if score > best_score:
            best_score = score
            best_plate = text

    if best_plate is None:
        return None, 0.0

    # 最终置信度 = 平均置信度
    final_conf = plate_scores[best_plate]['total_conf'] / plate_scores[best_plate]['count']
    return best_plate, min(final_conf, 1.0)


def recognize_plate(img, crop=None):
    """多级车牌识别策略（含图像增强 + 后处理校验）"""
    from improve_accuracy import enhance_image_for_plate, enhance_plate_region

    plateRecog = _models['plate_recog']

    # 策略1: 原始全图
    strategies = [
        ("full", lambda: plateRecog(img)),
    ]

    # 策略1b: 增强后的全图
    enhanced_full = enhance_image_for_plate(img)
    if enhanced_full is not None:
        strategies.append(("enhanced_full", lambda: plateRecog(enhanced_full)))

    # 策略2: 全图放大
    h_o, w_o = img.shape[:2]
    if h_o < 1000 or w_o < 1000:
        scale = min(2.0, 2000 / max(h_o, w_o))
        enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        strategies.append(("scaled", lambda: plateRecog(enlarged)))
        enhanced_enlarged = enhance_image_for_plate(enlarged)
        if enhanced_enlarged is not None:
            strategies.append(("enhanced_scaled", lambda: plateRecog(enhanced_enlarged)))

    # 策略3: 裁剪图
    if crop is not None and crop.size > 0:
        strategies.append(("crop", lambda: plateRecog(crop)))
        enhanced_crop = enhance_image_for_plate(crop)
        if enhanced_crop is not None:
            strategies.append(("enhanced_crop", lambda: plateRecog(enhanced_crop)))

        # 策略4: 裁剪图下半部分（车牌位置）
        h_c, w_c = crop.shape[:2]
        bottom = crop[int(h_c*0.55):int(h_c*0.95), int(w_c*0.1):int(w_c*0.9)]
        if bottom.size > 0:
            bottom = cv2.resize(bottom, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            strategies.append(("bottom", lambda: plateRecog(bottom)))
            enhanced_bottom = enhance_image_for_plate(bottom)
            if enhanced_bottom is not None:
                strategies.append(("enhanced_bottom", lambda: plateRecog(enhanced_bottom)))
            plate_binary = enhance_plate_region(bottom)
            if plate_binary is not None:
                strategies.append(("plate_binary", lambda: plateRecog(plate_binary)))

    all_results = []
    for name, fn in strategies:
        try:
            result = fn()
            if result and len(result) > 0:
                weight = 0.95 if name.startswith("enhanced_") or name == "plate_binary" else 1.0
                for r in result:
                    if len(r) > 1:
                        conf = float(r[1]) * weight if len(r) > 1 else 0
                        all_results.append((str(r[0]), conf, name))
        except Exception:
            continue

    if not all_results:
        raise PlateRecognizeError("所有车牌识别策略均失败")

    # 使用后处理投票融合
    best_plate, best_conf = _postprocess_plate_results(all_results)

    if best_plate is None:
        raise PlateRecognizeError("未能通过车牌格式校验")

    return best_plate, best_conf


def _cv2_to_pil(cv_img):
    """OpenCV BGR 转 PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _pil_to_cv2(pil_img):
    """PIL RGB 转 OpenCV BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_text_pil(pil_img, text, pos, font_size=20, color=(0, 255, 0)):
    """使用 PIL 绘制中文文本"""
    draw = ImageDraw.Draw(pil_img)
    if _FONT_PATH and os.path.exists(_FONT_PATH):
        font = ImageFont.truetype(_FONT_PATH, font_size)
    else:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return pil_img


def draw_result(img, detections, selected_idx, plate, v_type, v_color, brand, is_fake):
    """在图片上绘制检测框和结果标签（支持中文）"""
    vis = img.copy()
    h, w = vis.shape[:2]

    # 先画所有 OpenCV 元素（框）
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0) if i == selected_idx else (128, 128, 128)
        thickness = 3 if i == selected_idx else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    # 转为 PIL 绘制中文标签
    pil_img = _cv2_to_pil(vis)

    for i, det in enumerate(detections):
        x1, y1 = det['bbox'][:2]
        color = (0, 255, 0) if i == selected_idx else (128, 128, 128)
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # 文字背景
        draw = ImageDraw.Draw(pil_img)
        if _FONT_PATH and os.path.exists(_FONT_PATH):
            font = ImageFont.truetype(_FONT_PATH, 18)
        else:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        bg_color = tuple(color)
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 6, y1], fill=bg_color)
        draw.text((x1 + 3, y1 - th - 6), label, font=font, fill=(255, 255, 255))

    # 底部信息栏
    info_lines = [
        f"车牌: {plate}",
        f"类型: {v_type} | 颜色: {v_color}",
        f"品牌: {brand}",
        f"结果: {'套牌车' if is_fake else '正常'}"
    ]
    bar_h = 32 * len(info_lines) + 16
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw_o = ImageDraw.Draw(overlay)
    draw_o.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0, 180))
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')

    draw = ImageDraw.Draw(pil_img)
    if _FONT_PATH and os.path.exists(_FONT_PATH):
        font = ImageFont.truetype(_FONT_PATH, 22)
    else:
        font = ImageFont.load_default()

    for i, line in enumerate(info_lines):
        y = h - bar_h + 10 + i * 30
        if is_fake and i == len(info_lines) - 1:
            text_color = (255, 0, 0)
        else:
            text_color = (0, 255, 0)
        draw.text((20, y), line, font=font, fill=text_color)

    return _pil_to_cv2(pil_img)


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
        v_type, type_conf, v_color, color_conf = classify_crop(crop, models, full_img=img, bbox=best_det['bbox'])

        # 5. 品牌（优先使用独立品牌分类器，回退到YOLO）
        brand_model = models.get('brand_model')
        if brand_model is not None and models.get('brand_transform') is not None:
            # 使用独立ResNet18品牌分类器
            try:
                brand_input = models['brand_transform'](crop).unsqueeze(0).to(models['device'])
                with torch.no_grad():
                    brand_output = brand_model(brand_input)
                    brand_probs = torch.softmax(brand_output, dim=1)[0].cpu().numpy()
                brand_idx = int(np.argmax(brand_probs))
                brand = models['brand_classes'][brand_idx]
                brand_conf = float(brand_probs[brand_idx])
                # 如果置信度太低，标记为未知
                if brand_conf < 0.3:
                    brand = '未知'
                    brand_conf = 0.0
            except Exception as e:
                print(f"[WARN] 品牌分类器推理失败: {e}")
                brand = '未知'
                brand_conf = 0.0
        else:
            # 回退到YOLO品牌输出
            raw_brand = best_det['class_name']
            brand_conf = best_det['confidence']
            if brand_conf < 0.3 or raw_brand in ['其他', '黄牌大巴', '黄牌卡车', '公交车']:
                brand = '未知'
                brand_conf = 0.0
            else:
                brand = correct_brand(raw_brand, vehicle_type=v_type, vehicle_color=v_color)

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
        is_fake, true_brand, _ = check_fake_plate(plate, brand, detected_brand_conf=brand_conf)

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
