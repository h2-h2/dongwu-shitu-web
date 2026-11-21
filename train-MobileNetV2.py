import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# =============================================================================
# 0. 参数配置
# =============================================================================
TRAIN_DIR = 'dataset/train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
DROPOUT_RATE = 0.5
L2_REG = 0.001

PHASE1_EPOCHS = 30
PHASE1_LR = 0.001
PHASE2_EPOCHS = 60
PHASE2_LR = 0.0001

# =============================================================================
# 0.1 环境检查
# =============================================================================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"训练目录不存在: {TRAIN_DIR}")

os.makedirs('models', exist_ok=True)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU 可用: {len(gpus)} 个")
print(f"训练数据目录: {os.path.abspath(TRAIN_DIR)}")

# =============================================================================
# 1. 数据准备
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

print(f"训练样本数: {train_generator.samples}, 验证样本数: {val_generator.samples}")
print(f"类别映射: {train_generator.class_indices}")

# =============================================================================
# 2. 模型构建
# =============================================================================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, kernel_regularizer=regularizers.l2(L2_REG))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(L2_REG))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(DROPOUT_RATE)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================================================================
# 3. 第一阶段回调
# =============================================================================
phase1_early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

phase1_checkpoint = ModelCheckpoint(
    'models/mobilenetv2_grains_phase1_best.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

phase1_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
)

# =============================================================================
# 4. 第一阶段训练
# =============================================================================
print("=" * 60)
print("第一阶段训练：冻结骨干，只训练分类头")
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
print("第二阶段训练：解冻骨干最后5层，小学习率微调")
print("=" * 60)

base_model.trainable = True

# 冻结所有 BatchNormalization 层
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

# 解冻最后5层（非BN层）
for layer in base_model.layers[-5:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase2_early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

phase2_checkpoint = ModelCheckpoint(
    'models/mobilenetv2_grains.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

phase2_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1
)

phase2_initial_epoch = len(history_phase1.epoch)
history_phase2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    initial_epoch=phase2_initial_epoch,
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
with open('models/mobilenetv2_grains.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite 模型已保存: models/mobilenetv2_grains.tflite")
print(f"模型体积约: {len(tflite_model) / (1024 * 1024):.1f} MB")

print("\n" + "=" * 60)
print("训练完成！")
print("第一阶段最佳权重: models/mobilenetv2_grains_phase1_best.weights.h5")
print("最终权重（微调后）: models/mobilenetv2_grains.weights.h5")
print("TFLite 模型: models/mobilenetv2_grains.tflite")
print("=" * 60)
