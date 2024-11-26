from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_train_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE
from tensorflow.python.client import device_lib
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" #주석 풀어서 gpu 사용

INPUT_SIZE = DEFAULT_SIZE    #TODO 모델에 따라 INPUT_SIZE 조정
BATCH_SIZE = 64
EPOCHS = 15

# 하이퍼파라미터 그리드
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2] #필요하다면 추가적으로 hyperparameter 조정
optimizers = ['adam', 'rmsprop']

# Data Load
train_dataset, validation_dataset = load_train_data(img_size=INPUT_SIZE, gray=False, normalization=True, flatten=False, batch_size=BATCH_SIZE)

# 모델에 따라 추가적인 preprocessing 필요한 경우 있음 


# feature_extractor = tf.keras.applications.DenseNet169(
#     input_shape=(INPUT_SIZE,INPUT_SIZE, 3),
#     include_top=False,
#     weights="imagenet"
# )

# 모델 정의
model = models.Sequential([

    #TODO 모델 추가 필요
    
    
    # ex) DenseNet169
    # feature_extractor,
    # tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    # tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification"),
])

# Focal Loss
def focal_loss_sparse(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_pred_probs = tf.gather(y_pred, y_true[..., None], axis=-1, batch_dims=1)
        y_pred_probs = tf.squeeze(y_pred_probs, axis=-1)
        cross_entropy = -tf.math.log(y_pred_probs)
        p_t = tf.math.exp(-cross_entropy)
        loss = alpha * (1 - p_t) ** gamma * cross_entropy
        return tf.reduce_mean(loss)
    return loss

best_val_accuracy = 0
best_model = None
best_history = None
best_params = None

# Grid Search
for lr in learning_rates:
    for opt_name in optimizers:
        print(f"\n시도: learning_rate={lr}, optimizer={opt_name}")
        
        if opt_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        
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
            train_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # validation 데이터에서 정확도 계산
        val_metrics = model.evaluate(validation_dataset)
        val_accuracy = val_metrics[1]  # accuracy는 두 번째 메트릭
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_history = history.history
            best_params = {'learning_rate': lr, 'optimizer': opt_name}

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
for images, labels in validation_dataset:
    predictions = best_model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels, axis=1))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()