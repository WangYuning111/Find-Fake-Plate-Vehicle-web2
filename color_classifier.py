#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 HSV 色彩空间的车辆颜色分类器
- 无需训练，直接分析车身主色调
- 支持 10 种颜色：黑、白、银、灰、红、蓝、绿、黄、棕、橙
- 对光照变化比 RGB 更鲁棒
"""
import cv2
import numpy as np

# HSV 颜色范围定义 (H: 0-179, S: 0-255, V: 0-255)
# 格式: [(name, lower, upper, priority), ...]
# priority 越高，该颜色在重叠区域越有优先权
COLOR_RANGES = [
    ("white",   np.array([0,   0,   200]), np.array([179, 30,  255]), 5),
    ("silver",  np.array([0,   0,   140]), np.array([179, 30,  199]), 4),
    ("gray",    np.array([0,   0,   60]),  np.array([179, 40,  139]), 3),
    ("black",   np.array([0,   0,   0]),   np.array([179, 255, 80]),  6),
    ("red1",    np.array([0,   50,  50]),  np.array([10,  255, 255]), 6),
    ("red2",    np.array([160, 50,  50]),  np.array([179, 255, 255]), 6),
    ("orange",  np.array([11,  50,  50]),  np.array([25,  255, 255]), 5),
    ("yellow",  np.array([26,  50,  50]),  np.array([35,  255, 255]), 5),
    ("green",   np.array([36,  50,  50]),  np.array([85,  255, 255]), 5),
    ("blue",    np.array([86,  50,  50]),  np.array([125, 255, 255]), 5),
    ("brown",   np.array([10,  40,  30]),  np.array([30,  180, 120]), 3),
]

# 中文映射
COLOR_NAMES_ZH = {
    "black": "黑色", "white": "白色", "silver": "银色", "gray": "灰色",
    "red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色",
    "brown": "棕色", "orange": "橙色", "unknown": "未知"
}


def _get_dominant_color_mask(hsv_img, exclude_black=True, exclude_white=True):
    """
    获取图片中车身主体颜色的掩码
    - 排除极端阴影和极端高光
    - 保留黑色车身像素（V在15-80之间）
    """
    h, w = hsv_img.shape[:2]

    # 基本掩码：排除极低饱和度（接近灰色/黑白），但保留黑色车身
    mask = hsv_img[:, :, 1] > 10

    if exclude_black:
        # 只排除极端暗部（V<15），保留黑色车身像素
        mask &= hsv_img[:, :, 2] > 15
    if exclude_white:
        # 排除过亮（高光/天空反射）
        mask &= hsv_img[:, :, 2] < 245

    return mask


def classify_color_hsv(image_bgr, region_of_interest=None):
    """
    对车辆图片进行颜色分类（改进版：增强黑色/深蓝区分）
    :param image_bgr: OpenCV BGR 格式图片
    :param region_of_interest: 可选的感兴趣区域 (x1,y1,x2,y2)
    :return: (color_en, color_zh, confidence)
    """
    if image_bgr is None or image_bgr.size == 0:
        return "unknown", "未知", 0.0

    # 裁剪 ROI（如果提供）
    if region_of_interest is not None:
        x1, y1, x2, y2 = region_of_interest
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            image_bgr = image_bgr[y1:y2, x1:x2]

    # 转为 HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 缩小图片以加速处理
    scale = 0.3
    small = cv2.resize(hsv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 车身区域掩码（排除极端阴影/高光，但保留黑色车身）
    body_mask = _get_dominant_color_mask(small, exclude_black=True, exclude_white=True)
    body_pixels = np.count_nonzero(body_mask)

    if body_pixels < 100:
        body_mask = _get_dominant_color_mask(small, exclude_black=False, exclude_white=False)
        body_pixels = np.count_nonzero(body_mask)

    # 统计各颜色像素数
    color_votes = {}

    for name, lower, upper, priority in COLOR_RANGES:
        color_mask = cv2.inRange(small, lower, upper)
        color_mask &= body_mask.astype(np.uint8)
        count = np.count_nonzero(color_mask)
        score = count * priority
        base_name = "red" if name.startswith("red") else name
        if base_name not in color_votes:
            color_votes[base_name] = 0
        color_votes[base_name] += score

    # ========== 黑色 vs 深蓝 智能区分 ==========
    # 统计暗色像素 (V < 90) 在车身区域中的占比和饱和度
    dark_mask = (small[:, :, 2] < 90) & (small[:, :, 2] > 15)  # V在15-90之间
    dark_pixels = np.count_nonzero(dark_mask)

    if dark_pixels > 0 and body_pixels > 0:
        dark_ratio = dark_pixels / body_pixels
        sat_in_dark = small[:, :, 1][dark_mask]
        avg_sat_dark = np.mean(sat_in_dark)

        # 暗色像素占比高且平均饱和度低 -> 黑色车身
        if dark_ratio > 0.20 and avg_sat_dark < 90:
            # 强黑色信号，大幅提升黑色得分
            boost = int(body_pixels * 0.6)
            color_votes["black"] = color_votes.get("black", 0) + boost
        # 暗色像素占比高但饱和度高 -> 可能是深蓝
        elif dark_ratio > 0.20 and avg_sat_dark >= 120:
            boost = int(body_pixels * 0.3)
            color_votes["blue"] = color_votes.get("blue", 0) + boost

    if not color_votes or body_pixels < 50:
        return "unknown", "未知", 0.0

    total_score = sum(color_votes.values())
    if total_score == 0:
        return "unknown", "未知", 0.0

    best_color = max(color_votes, key=color_votes.get)
    best_score = color_votes[best_color]
    confidence = best_score / total_score

    if confidence < 0.25:
        return "unknown", "未知", confidence

    return best_color, COLOR_NAMES_ZH.get(best_color, best_color), confidence


def classify_color_with_crop(image_bgr, bbox=None):
    """
    更智能的颜色分类：排除车牌区域，只分析车身主体
    :param bbox: 车辆检测框 (x1,y1,x2,y2)，如果为 None 则分析全图
    """
    h, w = image_bgr.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        # 缩小检测框到车身主体（排除车牌、车轮区域）
        # 车牌通常在下方，所以上方 80% 更可能是车身
        body_y1 = y1 + int((y2 - y1) * 0.15)  # 去掉顶部一点
        body_y2 = y2 - int((y2 - y1) * 0.25)  # 去掉底部（车牌区域）
        body_x1 = x1 + int((x2 - x1) * 0.15)  # 去掉左侧
        body_x2 = x2 - int((x2 - x1) * 0.15)  # 去掉右侧

        body_x1 = max(0, body_x1)
        body_y1 = max(0, body_y1)
        body_x2 = min(w, body_x2)
        body_y2 = min(h, body_y2)

        return classify_color_hsv(image_bgr, (body_x1, body_y1, body_x2, body_y2))
    else:
        return classify_color_hsv(image_bgr)


# 兼容旧的 API
def predict_color(image_bgr, bbox=None):
    """兼容旧接口的颜色预测"""
    color_en, color_zh, conf = classify_color_with_crop(image_bgr, bbox)
    return color_en, conf


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python color_classifier.py <图片路径>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"无法读取图片: {sys.argv[1]}")
        sys.exit(1)

    color_en, color_zh, conf = classify_color_hsv(img)
    print(f"识别颜色: {color_en} ({color_zh}), 置信度: {conf:.2%}")
