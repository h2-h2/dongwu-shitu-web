import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# =============================================================================
# 0. 参数配置
# =============================================================================
# 随机种子：固定随机性，保证每次运行结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 路径配置
TRAIN_DIR = 'dataset/train'          # 训练数据目录，内含 5 个子目录（每类一个）
MODEL_DIR = 'models'                 # 模型保存目录
MODEL_NAME = 'efficientnetb0_grains' # 模型文件名前缀

# 训练超参数
IMG_SIZE = (224, 224)  # EfficientNetB0 标准输入尺寸
BATCH_SIZE = 8          # 小 batch，适合小样本场景（配合 BN 冻结策略）
DROPOUT_RATE = 0.5      # Dropout 比例，抑制过拟合
L2_REG = 0.001          # L2 正则化系数，约束全连接层权重

# 第一阶段：冻结骨干，只训练分类头
PHASE1_EPOCHS = 30      # 第一阶段训练轮数（训练分类头）
PHASE1_LR = 0.001       # 初始学习率（分类头随机初始化，需要较大学习率）

# 第二阶段：解冻部分骨干，小学习率微调
PHASE2_EPOCHS = 60      # 第二阶段训练轮数（微调骨干）
PHASE2_LR = 1e-5        # 微调学习率（极低，保护预训练特征不被破坏）
UNFREEZE_RATIO = 0.2    # 解冻比例：只解冻最后 20% 的层

# =============================================================================
# 0.1 环境检查
# =============================================================================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"训练目录不存在: {TRAIN_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)  # 确保模型保存目录存在

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU 可用: {len(gpus)} 个")
print(f"训练数据目录: {os.path.abspath(TRAIN_DIR)}")

# =============================================================================
# 1. 数据准备
# =============================================================================
# 训练集生成器：随机旋转、平移、亮度、剪切、缩放、水平翻转
train_datagen = ImageDataGenerator(
    rescale=1./255,                    # 像素归一化到 [0, 1]
    rotation_range=30,                 # 随机旋转 ±30°
    width_shift_range=0.25,            # 水平平移 ±25% 图像宽度
    height_shift_range=0.25,           # 垂直平移 ±25% 图像高度
    brightness_range=[0.8, 1.2],       # 随机亮度调整
    shear_range=0.2,                   # 剪切变换
    zoom_range=0.2,                    # 随机缩放
    horizontal_flip=True,              # 水平翻转（动物图像水平翻转不影响类别）
    fill_mode='nearest',               # 边界填充方式
    validation_split=0.2               # 20% 数据用于验证
)

# 验证集生成器：只做 rescale，不做任何增强
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 从目录读取数据，自动按子目录名生成标签
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,       # 统一缩放到 224×224
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # 多分类 one-hot 编码
    subset='training',          # 80% 作为训练集
    seed=SEED                   # 固定划分，保证可复现
)

val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',        # 20% 作为验证集
    seed=SEED,                  # 与训练集相同 seed，保证划分一致
    shuffle=False               # 验证集不 shuffle，保证每次评估结果一致
)

# 动态获取类别数（从目录结构自动推断，无需硬编码）
NUM_CLASSES = len(train_generator.class_indices)
print(f"检测到 {NUM_CLASSES} 个类别: {train_generator.class_indices}")

# =============================================================================
# 2. 模型构建
# =============================================================================
# 加载预训练骨干（ImageNet 权重，不含分类头）
base_model = EfficientNetB0(
    weights='imagenet',        # 使用 ImageNet 预训练权重
    include_top=False,         # 不包含原分类头（1000 类）
    input_shape=(*IMG_SIZE, 3) # 输入尺寸 224×224×3
)

# 构建分类头
x = base_model.output
# GAP：将空间维度压缩为单值，大幅减少参数，天然抗过拟合
x = GlobalAveragePooling2D()(x)
# Dense(128) + L2：小样本下适中的隐藏层维度，L2 正则约束权重幅度
x = Dense(128, kernel_regularizer=regularizers.l2(L2_REG))(x)
# BN → ReLU：先归一化再激活，避免 BN 与 Dropout 的方差偏移问题
x = BatchNormalization()(x)
x = Activation('relu')(x)
# Dropout：训练时随机丢弃 50% 神经元，测试时自动关闭
x = Dropout(DROPOUT_RATE)(x)
# Softmax 输出 N 类概率
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 第一阶段：冻结整个骨干网络
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss='categorical_crossentropy',  # 多分类交叉熵损失
    metrics=['accuracy']
)

model.summary()

# =============================================================================
# 3. 第一阶段回调
# =============================================================================
# EarlyStopping：验证损失不再下降时提前停止，防止过拟合，回滚到最佳权重
phase1_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,               # 连续 5 轮不下降则停止
    restore_best_weights=True, # 回滚到验证损失最低的权重
    verbose=1
)

# ModelCheckpoint：只保存权重（更轻量，便于迁移）
phase1_checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, f'{MODEL_NAME}_phase1_best.weights.h5'),
    monitor='val_accuracy',
    save_best_only=True,      # 只保存最佳权重
    save_weights_only=True,   # 仅保存权重，不保存完整模型
    mode='max',
    verbose=1
)

# ReduceLROnPlateau：验证损失停滞时学习率减半，帮助跳出局部最优
phase1_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,               # 学习率乘以 0.5
    patience=5,               # 连续 5 轮不下降则降低
    min_lr=1e-7,              # 学习率下限
    verbose=1
)

# =============================================================================
# 4. 第一阶段训练
# =============================================================================
print("=" * 60)
print("第一阶段训练：冻结骨干，只训练分类头")
print(f"训练样本: {train_generator.samples}, 验证样本: {val_generator.samples}")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=val_generator,
    callbacks=[phase1_early_stopping, phase1_checkpoint, phase1_reduce_lr],
    verbose=1
)

# =============================================================================
# 5. 第二阶段训练
# =============================================================================
print("\n" + "=" * 60)
print("第二阶段训练：解冻骨干最后20%层，小学习率微调")
print("=" * 60)

# 启用骨干训练
base_model.trainable = True

# 冻结所有 BN 层（核心：小 batch 下 BN 统计量不可靠，必须冻结）
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

# 按比例解冻：只让最后 UNFREEZE_RATIO（20%）的层参与训练
total_layers = len(base_model.layers)
unfreeze_from = int(total_layers * (1 - UNFREEZE_RATIO))
for layer in base_model.layers[:unfreeze_from]:
    layer.trainable = False

# 统计可训练层数，便于确认配置正确
trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"骨干总层数: {total_layers}, 解冻层数: {trainable_count} (从第{unfreeze_from}层开始)")

# 重新编译：使用极低学习率，保护预训练特征
model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 第二阶段回调
phase2_early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

phase2_checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, f'{MODEL_NAME}.weights.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,   # 仅保存权重
    mode='max',
    verbose=1
)

phase2_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1
)

# 第二阶段独立训练，不使用 initial_epoch
# 原因：第二阶段重新编译了模型，optimizer 状态已重置，initial_epoch 无实际意义
history_phase2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    validation_data=val_generator,
    callbacks=[phase2_early_stopping, phase2_checkpoint, phase2_reduce_lr],
    verbose=1
)

# =============================================================================
# 6. 转换为 TensorFlow Lite 格式（减小模型体积）
# =============================================================================
print("\n" + "=" * 60)
print("转换为 TensorFlow Lite 格式...")
print("=" * 60)

# 转换为 TFLite（默认优化）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存 TFLite 模型
tflite_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite 模型已保存: {tflite_path}")
print(f"模型体积约: {len(tflite_model) / (1024 * 1024):.1f} MB")

print("\n" + "=" * 60)
print("训练完成！")
print(f"第一阶段最佳权重: {MODEL_DIR}/{MODEL_NAME}_phase1_best.weights.h5")
print(f"最终权重（微调后）: {MODEL_DIR}/{MODEL_NAME}.weights.h5")
print(f"TFLite 模型: {MODEL_DIR}/{MODEL_NAME}.tflite")
print("=" * 60)
