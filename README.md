# 套牌车智能稽查系统 V2.0

基于深度学习的车辆识别与套牌检测平台。

## 功能特性

- **车辆检测**：YOLOv12 实时检测车辆位置
- **车牌识别**：HyperLPR 多级策略识别中文车牌
- **属性识别**：车型（4类）+ 颜色（8色）分类
- **套牌判定**：车牌-品牌数据库比对
- **结果可视化**：检测框 + 标签叠加在结果图上
- **历史记录**：SQLite 持久化存储，支持分页查询
- **车辆库管理**：支持增删改查车辆信息
- **批量上传**：支持同时分析多张图片
- **RESTful API**：完整的 JSON 接口

## 快速启动

### 方式一：直接运行（开发环境）

```bash
pip install -r requirements.txt
python app.py
```

访问 http://localhost:8090

### 方式二：Docker 部署（生产环境）

```bash
docker-compose up -d
```

## 项目结构

```
.
├── app.py                  # Flask 应用主入口
├── config.py               # 配置管理
├── database.py             # SQLite 数据库操作
├── inference.py            # 模型推理封装
├── train_vehicle_type.py   # 车型分类训练
├── train_vehicle_brand.py  # 品牌分类训练
├── train_yolo.py           # YOLO fine-tune
├── cfg/                    # 模型权重
├── data/                   # SQLite 数据库
├── static/
│   ├── uploads/            # 上传图片
│   └── results/            # 结果图片（带标注）
├── templates/              # HTML 模板
└── datasets/               # 训练数据集
```

## API 文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/analyze` | POST/GET | 单张图片分析 |
| `/api/batch_analyze` | POST | 批量图片分析 |
| `/api/records` | GET | 查询识别记录 |
| `/api/vehicles` | GET/POST | 车辆列表/添加 |
| `/api/vehicles/<plate>` | PUT/DELETE | 更新/删除车辆 |
| `/api/stats` | GET | 统计信息 |
| `/health` | GET | 健康检查 |
| `/warmup` | GET | 模型预热 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FLASK_DEBUG` | False | 调试模式 |
| `PORT` | 8090 | 服务端口 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `TORCH_DEVICE` | auto | cuda / cpu |
| `YOLO_CONF` | 0.25 | YOLO 置信度阈值 |

## 训练提升精度

1. 下载数据集（见 `datasets/README.md`）
2. 运行 `prepare_dataset.py` 整理数据
3. 训练模型：
   - `python train_yolo.py` — 提升检测精度
   - `python train_vehicle_type.py` — 提升车型分类
   - `python train_vehicle_brand.py` — 提升品牌识别
4. 替换 `cfg/` 下的模型权重

## 许可证

MIT
