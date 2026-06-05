# 训练数据集获取指南

## 一、品牌识别数据集（YOLO fine-tune / 分类训练）

### 推荐 1：车辆品牌与类型检测数据集（YOLO格式）
- **规模**：7,029 张图像，14,058 个标注框
- **类别**：多种品牌 + 类型
- **格式**：YOLO txt 标注，可直接训练
- **获取**：https://www.2zcode.com/2282.html

### 推荐 2：Stanford Cars（细粒度品牌型号分类）
- **规模**：16,185 张，196 类汽车型号
- **用途**：训练独立的品牌分类模型
- **获取**：https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
- **说明**：需要将196类映射为你需要的品牌（如大众、宝马、丰田等）

### 推荐 3：CompCars Dataset
- **规模**：约 13.6 万张车辆图像
- **类别**：1,716 种车型，163 种汽车品牌
- **特点**：包含车辆前部、后部、侧面多角度
- **获取**：http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- **推荐度**：⭐⭐⭐⭐⭐（最适合做品牌识别）

---

## 二、车型分类数据集（4类：bus/car/minibus/truck）

### 推荐 1：UA-DETRAC
- **规模**：14 万帧，8,250 辆车
- **类别**：car, bus, van, others
- **场景**：真实交通监控
- **获取**：http://detrac-db.rit.albany.edu/
- **用法**：van 可归为 minibus，others 过滤掉

### 推荐 2：BITVehicle Dataset
- **规模**：约 9,000 张
- **特点**：包含车型和颜色标注
- **获取**：http://iitlab.bit.edu.cn/mcislab/vehicledb/

### 推荐 3：自定义组合方案
从以下数据集中提取对应类别：
- **bus**：UA-DETRAC bus + COCO bus
- **car**：UA-DETRAC car + Stanford Cars
- **minibus**：UA-DETRAC van + 自行收集面包车图片
- **truck**：UA-DETRAC others中的卡车 + COCO truck

---

## 三、快速下载脚本

部分数据集需官网申请或网盘下载。建议优先下载 **CompCars**（品牌最全）和 **UA-DETRAC**（车型场景最真实）。
