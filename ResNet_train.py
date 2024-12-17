from tqdm import tqdm
import os
from PIL import Image
from loss_function import focal_loss_sparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_train_data, load_test_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #주석 풀어서 gpu 사용

INPUT_SIZE = DEFAULT_SIZE    
EPOCHS = 10
BATCH_SIZE = 32

# 하이퍼파라미터 그리드
learning_rates = [1e-4]  
optimizers = ['adam']
activation_functions = ['elu'] 

# Data Load
train_ds, val_ds = load_train_data(img_size=INPUT_SIZE, gray=False, batch_size = BATCH_SIZE)

# Test Data Load
#test_dataset = load_test_data(img_size=INPUT_SIZE, gray=False, batch_size=BATCH_SIZE)



best_val_accuracy = 0
best_model = None
best_history = None
best_params = None

# Grid Search
for lr in learning_rates:
    for opt_name in optimizers:
        for activation_name in activation_functions:
            
            
            # 각 반복마다 새로운 모델 생성
            model = tf.keras.models.Sequential([
                tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=[INPUT_SIZE, INPUT_SIZE, 3]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(7, activation='softmax')
            ])



            # optimizer 설정
            if opt_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            
            # 모델 컴파일, 손실함수 설정
            model.compile(
                optimizer=optimizer,
                loss = "sparse_categorical_crossentropy",
                metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                './models/ResNet.keras',  # 모델 저장 경로, model 이름으로
                monitor='val_accuracy',
                save_best_only=True,  # 가장 좋은 모델만 저장
                mode='max',  # val_accuracy가 최대일 때 저장
                verbose=1
            )
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks=[early_stopping,model_checkpoint],
                verbose=1,
            )
            
            # validation 데이터에서 정확도 계산
            val_metrics = model.evaluate(val_ds)
            val_accuracy = val_metrics[1]  # accuracy는 두 번째 메트릭
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_history = history.history
                
               
                best5params = {'learning_rate': lr, 'optimizer': opt_name, 'activation': activation_name}

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

# Top-2 Accuracy 그래프
plt.figure(figsize=(10, 6))
plt.plot(best_history['sparse_top_k_categorical_accuracy'], label='Training Top-2 Accuracy')
plt.plot(best_history['val_sparse_top_k_categorical_accuracy'], label='Validation Top-2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Top-2 Accuracy')
plt.title('Training and Validation Top-2 Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix 계산을 위한 수정
y_true = []
y_pred = []


# 전체 validation 데이터셋에서 예측값과 실제값 수집
for images, labels in val_ds:
    predictions = best_model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()