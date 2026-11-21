# 模型加载和预测（仅支持 TFLite 格式）
import numpy as np
import os
import json
import random
from django.conf import settings

# 中文标签映射（英文代码 → 中文名）
CLASS_NAMES_CN = {
    'ji': '鸡',
    'niu': '牛',
    'yang': '羊',
    'ya': '鸭',
    'zhu': '猪'
}

# 默认类别顺序（仅当 class_indices.json 不存在时回退使用）
_DEFAULT_CLASS_NAMES = ['ji', 'niu', 'ya', 'yang', 'zhu']


def _class_indices_path():
    """返回 class_indices.json 的路径（由训练脚本生成，重训后才会存在）"""
    return os.path.join(settings.BASE_DIR, 'models', 'class_indices.json')


def _load_class_names():
    """
    从 class_indices.json 读取类别顺序，避免推理时索引错位。
    若文件不存在（未重训），回退到默认顺序。
    """
    path = _class_indices_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                idx_map = json.load(f)  # 形如 {'ji':0,'niu':1,'ya':2,'yang':3,'zhu':4}
            # 按索引升序排列得到类别名列表
            return [k for k, v in sorted(idx_map.items(), key=lambda x: x[1])]
        except Exception:
            pass
    return list(_DEFAULT_CLASS_NAMES)


# 类别顺序：重训后从 json 读取，否则用默认
CLASS_NAMES = _load_class_names()

# 全局变量，用于跟踪模型加载状态
MODEL_LOADED = False  # 标记模型是否已成功加载
interpreter1 = None    # MobileNetV2 TFLite 解释器
interpreter2 = None    # EfficientNetB0 TFLite 解释器
input_details1 = None  # MobileNetV2 输入详情
input_details2 = None  # EfficientNetB0 输入详情
output_details1 = None # MobileNetV2 输出详情
output_details2 = None # EfficientNetB0 输出详情


def _load_tflite_model(model_path):
    """加载 TFLite 模型"""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def _load_models():
    """
    懒加载双模型（仅支持 TFLite 格式）
    """
    global MODEL_LOADED, interpreter1, interpreter2, input_details1, input_details2, output_details1, output_details2

    # 检查模型是否已加载
    if MODEL_LOADED and interpreter1 is not None and interpreter2 is not None:
        import tensorflow as tf
        return tf

    try:
        import tensorflow as tf

        # 加载 TFLite 模型
        tflite_path1 = os.path.join(settings.BASE_DIR, 'models', 'mobilenetv2_grains.tflite')
        tflite_path2 = os.path.join(settings.BASE_DIR, 'models', 'efficientnetb0_grains.tflite')

        if os.path.exists(tflite_path1) and os.path.exists(tflite_path2):
            interpreter1, input_details1, output_details1 = _load_tflite_model(tflite_path1)
            interpreter2, input_details2, output_details2 = _load_tflite_model(tflite_path2)
            MODEL_LOADED = True
            return tf

        MODEL_LOADED = False
        interpreter1 = None
        interpreter2 = None

    except Exception:
        MODEL_LOADED = False
        interpreter1 = None
        interpreter2 = None

    return None


def _has_new_preprocess():
    """
    判断是否启用新预处理。
    当 class_indices.json 存在时，认为模型已用新预处理重训，
    推理时切换为与训练一致的预处理（MobileNetV2: preprocess_input；EfficientNet: [0,255]）。
    否则沿用旧的 /255 预处理，兼容旧模型。
    """
    return os.path.exists(_class_indices_path())


def preprocess_image(img_path):
    """
    预处理图像，为两个模型分别生成输入。

    - 新模式（class_indices.json 存在，模型已重训）：
        * MobileNetV2 → mobilenetv2.preprocess_input（[-1,1]），匹配预训练权重
        * EfficientNetB0 → 保持 [0,255] 原始像素（其内部归一化层会处理）
    - 旧模式（未重训，沿用旧模型）：
        * 两个模型都用 /255 归一化（[0,1]），与旧模型训练时一致
    """
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)            # (224,224,3), 值域 [0,255]
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    if _has_new_preprocess():
        from tensorflow.keras.applications import mobilenetv2
        # MobileNetV2：归一化到 [-1,1]，匹配预训练权重
        img_mobilenet = mobilenetv2.preprocess_input(img_array.copy()).astype(np.float32)
        # EfficientNetB0：保持 [0,255]，其内部归一化层会处理
        img_efficientnet = img_array.astype(np.float32)
    else:
        # 旧模式：与旧模型训练时的 /255 一致
        img_mobilenet = (img_array / 255.0).astype(np.float32)
        img_efficientnet = (img_array / 255.0).astype(np.float32)

    return img_mobilenet, img_efficientnet


def predict_image(file_path):
    """双模型预测函数（仅使用 TFLite）"""
    _load_models()

    if MODEL_LOADED and interpreter1 is not None and interpreter2 is not None:
        img_mobilenet, img_efficientnet = preprocess_image(file_path)

        # TFLite 模型预测
        interpreter1.set_tensor(input_details1[0]['index'], img_mobilenet)
        interpreter1.invoke()
        pred1 = interpreter1.get_tensor(output_details1[0]['index'])

        interpreter2.set_tensor(input_details2[0]['index'], img_efficientnet)
        interpreter2.invoke()
        pred2 = interpreter2.get_tensor(output_details2[0]['index'])

        label1 = CLASS_NAMES[np.argmax(pred1)]
        label2 = CLASS_NAMES[np.argmax(pred2)]
        confidence1 = float(np.max(pred1))
        confidence2 = float(np.max(pred2))
        prob1 = [float(p) for p in pred1[0]]
        prob2 = [float(p) for p in pred2[0]]
    else:
        # 模型未加载，返回随机结果
        label1 = random.choice(CLASS_NAMES)
        label2 = random.choice(CLASS_NAMES)
        confidence1 = round(random.uniform(0.7, 0.99), 2)
        confidence2 = round(random.uniform(0.7, 0.99), 2)
        prob1 = _random_probs(label1)
        prob2 = _random_probs(label2)

    return {
        'label1': CLASS_NAMES_CN.get(label1, label1),
        'label1_code': label1,
        'confidence1': confidence1,
        'label2': CLASS_NAMES_CN.get(label2, label2),
        'label2_code': label2,
        'confidence2': confidence2,
        'model_loaded': MODEL_LOADED,
        'prob1': prob1,
        'prob2': prob2,
    }


def _random_probs(main_label):
    """生成随机概率分布"""
    n = len(CLASS_NAMES)
    probs = [round(random.uniform(0.01, 0.1), 4) for _ in range(n)]
    idx = CLASS_NAMES.index(main_label)
    probs[idx] = round(random.uniform(0.6, 0.95), 4)
    total = sum(probs)
    probs = [round(p / total, 4) for p in probs]
    return probs
