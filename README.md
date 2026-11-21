# 动物识图 Web（dongwu-shitu-web）

基于深度学习的五类动物图像识别系统：**MobileNetV2 + EfficientNetB0 双模型**迁移学习对比，TFLite 轻量化，Django Web 上传识别与双模型结果并列对比。

> 本仓库为**开源代码版**：不包含训练数据集（448 张）与训练好的模型文件。复现识别效果需自备数据集并先运行训练脚本；**模型文件缺失时页面进入演示模式（返回随机结果，仅用于演示界面流程）**。

## 功能

- 上传图片 → 双模型预测 → 结果并列对比（中文标签 + 置信度）
- 五类动物：鸡 / 牛 / 羊 / 鸭 / 猪
- TFLite 默认优化转换，模型体积减小，适合 Web 端轻量部署

## 目录结构

```
dongwu-shitu-web/
├── manage.py
├── train_EfficientNetB0.py   # EfficientNetB0 两阶段迁移学习 + TFLite 转换
├── train-MobileNetV2.py      # MobileNetV2 两阶段迁移学习 + TFLite 转换
├── dw/                       # Django 项目配置
├── dw_app/                   # 识别应用（视图 / 表单 / 推理工具 / 模板）
├── dataset/train/            # 数据集（未包含，自备：ji / niu / ya / yang / zhu 五个子目录）
└── models/                   # 模型输出目录（未包含，由训练脚本生成）
```

## 训练设计要点

- **两阶段迁移学习**：先冻结骨干训练分类头，再解冻顶层以极低学习率（1e-5）微调
- EfficientNetB0 解冻最后 20% 层并**冻结全部 BN 层**（小 batch 下 BN 统计量不可靠）；MobileNetV2 只解冻最后 5 层，更保守
- 数据增强：旋转 / 平移 / 亮度 / 剪切 / 缩放 / 翻转，8:2 训练验证划分
- EarlyStopping + ModelCheckpoint + ReduceLROnPlateau 回调齐全；EfficientNetB0 脚本固定随机种子保证可复现
- 分类头：EfficientNetB0 单层 128；MobileNetV2 512 + 256 双层，均带 L2 正则化 + Dropout

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据集 `dataset/train/{ji,niu,ya,yang,zhu}/`（参考规模：每类 80–100 张）
3. 训练并生成 TFLite：`python train_EfficientNetB0.py` 与 `python train-MobileNetV2.py`
4. 启动：`python manage.py migrate && python manage.py runserver`
5. 打开 `http://127.0.0.1:8000`，上传图片查看双模型对比结果

## 说明

- 模型文件名沿用 `*_grains.tflite` 前缀（历史命名，不影响功能）
- `models/class_indices.json` 存在时，推理自动切换为与重训一致的预处理（MobileNetV2 归一化到 [-1,1]，EfficientNet 保持 [0,255]）；否则兼容旧的 /255 模型
- `SECRET_KEY` 从环境变量 `DJANGO_SECRET_KEY` 读取；生产部署请置 `DEBUG=False` 并配置 `ALLOWED_HOSTS`
