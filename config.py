#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置管理
所有硬编码参数集中到这里，支持环境变量覆盖
"""
import os

# ==================== 服务配置 ====================
class Config:
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fake-plate-dev-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 8090))
    HOST = os.environ.get('HOST', '0.0.0.0')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 最大上传 16MB

    # 文件上传
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')

    # 数据库
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'data/records.db')
    VEHICLE_CSV = os.environ.get('VEHICLE_CSV', 'vehicle-database.csv')

    # 模型路径
    YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'cfg/best.pt')
    VEHICLE_TYPE_MODEL_PATH = os.environ.get('TYPE_MODEL_PATH', 'cfg/vehicle_type.pth')
    VEHICLE_COLOR_MODEL_PATH = os.environ.get('COLOR_MODEL_PATH', 'cfg/vehicle_color.pth')

    # 推理参数
    YOLO_CONF = float(os.environ.get('YOLO_CONF', 0.25))      # YOLO 置信度阈值
    YOLO_IOU = float(os.environ.get('YOLO_IOU', 0.45))        # YOLO NMS IoU 阈值
    PLATE_CONF_THRESHOLD = float(os.environ.get('PLATE_CONF', 0.75))  # 车牌置信度阈值
    PLATE_MIN_LENGTH = int(os.environ.get('PLATE_MIN_LEN', 5))        # 车牌最小字符数

    # 设备
    DEVICE = os.environ.get('TORCH_DEVICE', 'auto')  # auto / cuda / cpu

    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        for d in [cls.UPLOAD_FOLDER, cls.RESULT_FOLDER,
                  os.path.dirname(cls.DATABASE_PATH)]:
            os.makedirs(d, exist_ok=True)
