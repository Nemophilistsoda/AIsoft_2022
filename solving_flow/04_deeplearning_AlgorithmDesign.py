import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import yaml
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# 加载配置文件
with open('F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\configs\\04_model_configs.yaml') as file:
    cofigs = yaml.safe_load(file)

def build_model():
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = inputs
    for i in range(5):
        filter = 2 ** (i + 6)
        x = tf.keras.layers.Conv2D(
            filter,
            kernel_size=(3, 3),
            activation='relu'
        )
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# 新增数据加载函数
def load_image_data(img_dir, target_size=(64, 64)):
    images = []
    for img_name in sorted(os.listdir(img_dir)):  # 按文件名排序保证顺序
        img_path = os.path.join(img_dir, img_name)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # 归一化
        images.append(img_array)
    return np.array(images)

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    return np.array(labels)

# 加载数据
img_dir = 'F:\\Competition_2025\\BDAIAC\\AIsoft_2022\data\\task4\\imgdata'
train_images = load_image_data(img_dir)  # 加载所有图片
train_labels = load_labels('F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\data\\task4\\trainlabels.txt')
test_labels = load_labels('F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\data\\task4\\testlabels.txt')

# 根据标签数量划分数据集
train_size = len(train_labels)
test_size = len(test_labels)
test_images = train_images[train_size:train_size+test_size]
train_images = train_images[:train_size]

model = build_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('✅Model training...')
history = model.fit(
    train_images,
    train_labels,
    epochs=cofigs['compiling']['epochs'],
    batch_size=cofigs['compiling']['batch_size'],
    verbose=1,
    validation_split=0.2  # 添加验证集分割
)

print('✅Model evaluation...')
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=0
)
print(f'\n测试集准确率：{test_accuracy:.4f}, 测试集损失：{test_loss:.4f}')