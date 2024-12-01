from tqdm import tqdm
import os
from PIL import Image
from loss_function import focal_loss_sparse
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_train_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 사용 설정 (필요시 주석 해제)

INPUT_SIZE = DEFAULT_SIZE  # TODO: AlexNet은 224x224의 입력 크기를 기본으로 함
EPOCHS = 10
BATCH_SIZE = 64

# 하이퍼파라미터 그리드
# TODO: 추가적으로 hyperparameter 조정
learning_rates = [1e-4, 1e-3, 1e-2]  
optimizers = ['adam', 'sgd_momentum']
activation_functions = ['relu', 'elu']  # TODO: activation function에 따라 kernel_initializer 조정 필요

# Data Load
train_dataset, validation_dataset = load_train_data(img_size=INPUT_SIZE, gray=True, normalization=True, flatten=False)

# TODO: 모델에 따라 추가적인 preprocessing 필요한 경우 있음, data_loader.py에 추가
# AlexNet의 경우 이미지를 회색조로 처리한 경우 입력 채널을 확인해야 합니다.

# 모델 정의
# TODO: 아래에 AlexNet 모델을 추가
def create_alexnet(input_size, activation_name):
    return models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation=activation_name, input_shape=(input_size, input_size, 1), kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),  # padding='same'으로 수정

        layers.Conv2D(256, (5, 5), activation=activation_name, padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),  # padding='same'으로 수정

        layers.Conv2D(384, (3, 3), activation=activation_name, padding='same', kernel_initializer='he_normal'),
        layers.Conv2D(384, (3, 3), activation=activation_name, padding='same', kernel_initializer='he_normal'),
        layers.Conv2D(256, (3, 3), activation=activation_name, padding='same', kernel_initializer='he_normal'),
        layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),  # padding='same'으로 수정

        layers.Flatten(),
        layers.Dense(4096, activation=activation_name, kernel_initializer='he_normal'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation=activation_name, kernel_initializer='he_normal'),
        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])


model = create_alexnet(INPUT_SIZE, 'relu')
model.summary()  # 모델 구조 확인

best_val_accuracy = 0
best_model = None
best_history = None
best_params = None

# Grid Search
for lr in learning_rates:
    for opt_name in optimizers:
        for activation_name in activation_functions:
            
            train_ds = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            val_ds = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            # 새 모델 생성
            model = create_alexnet(INPUT_SIZE, activation_name)
            
            # optimizer 설정
            if opt_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            
            # 모델 컴파일
            model.compile(
                optimizer=optimizer,
                loss=focal_loss_sparse(),
                metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # validation 데이터에서 정확도 계산
            val_metrics = model.evaluate(val_ds)
            val_accuracy = val_metrics[1]  # 두 번째 메트릭: accuracy
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_history = history.history
                best_params = {'learning_rate': lr, 'optimizer': opt_name, 'activation': activation_name}

print(f"\n최적 파라미터: {best_params}")
print(f"최고 검증 정확도: {best_val_accuracy:.4f}")

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(best_history['accuracy'], label='Training Accuracy')
plt.plot(best_history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(best_history['sparse_top_k_categorical_accuracy'], label='Training Top-2 Accuracy')
plt.plot(best_history['val_sparse_top_k_categorical_accuracy'], label='Validation Top-2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Top-2 Accuracy')
plt.title('Training and Validation Top-2 Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix 계산
y_true = []
y_pred = []

# 전체 validation 데이터셋에서 예측값과 실제값 수집
for images, labels in validation_dataset:
    predictions = best_model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())  # 원-hot encoding이 아닌 index 그대로 사용

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
