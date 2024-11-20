from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_loader import load_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE



INPUT_SIZE = DEFAULT_SIZE #48
BATCH_SIZE = 64
EPOCHS = 30

# Data Load
train_dataset, val_dataset = load_data("train", img_size=INPUT_SIZE, gray=True, normalization=True, flatten=False, augment=False, batch_size=BATCH_SIZE)

# 모델에 따라 추가적인 preprocessing 필요한 경우 있음 
# def preprocess_with_densenet(image, label):
#     image = tf.cast(image, tf.float32)  # 이미지 데이터를 float32로 변환
#     image = keras.applications.densenet.preprocess_input(image)    
#     return image, label

# train_dataset = train_dataset.map(preprocess_with_densenet)
# val_dataset = val_dataset.map(preprocess_with_densenet)

# feature_extractor = tf.keras.applications.DenseNet169(input_shape=(INPUT_SIZE,INPUT_SIZE, 3),
#                                                include_top=False,
#                                                weights="imagenet")

# 모델 정의
model = models.Sequential([
    # ex)
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 1)),
    # layers.MaxPooling2D((2, 2)),
    
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    
    # layers.Flatten(),
    
    # layers.Dense(128, activation='relu'),
    # layers.Dropout(0.5),
    
    # layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# EarlyStopping, 사용하려면, callbacks에
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Model Chekpoint (가장 좋은 모델 저장)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    './models/???.keras',  # 모델 저장 경로, model 이름으로
    monitor='val_accuracy',
    save_best_only=True,  # 가장 좋은 모델만 저장
    mode='max',  # val_accuracy가 최대일 때 저장
    verbose=1
)

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
)

# 모델 학습
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=1,
    callbacks=[model_checkpoint]
)

# 모델 예측 및 정확도 계산
y_true = []
y_pred = []

# train_dataset에서 예측
for images, labels in val_dataset:
    # 예측값 얻기
    batch_pred = model.predict(images)  # 예측값 (배치 단위)
    batch_pred = np.argmax(batch_pred, axis=1)  # 가장 높은 확률을 가진 클래스의 인덱스 추출
    
    # 실제 레이블 저장
    y_true.extend(labels.numpy())  # 실제 레이블을 numpy 배열로 변환해서 저장
    y_pred.extend(batch_pred)  # 예측값 저장

# accuracy_score 계산
final_accuracy = accuracy_score(y_true, y_pred)
print(f"Final training accuracy using sklearn's accuracy_score: {final_accuracy * 100:.2f}%")

# 학습 및 검증 정확도 그래프
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Validation Accuracy')
plt.legend()
plt.show()

