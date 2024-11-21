from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import sklearn
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data_loader import load_train_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE

INPUT_SIZE = DEFAULT_SIZE #48
BATCH_SIZE = 32
EPOCHS = 10

#hyperparameter tuning
#learning rate, optimizer, batch size

# Data Load
X, y, X_train, y_train, X_val, y_val = load_train_data(img_size=INPUT_SIZE, gray=True, normalization=True, flatten=False, batch_size=BATCH_SIZE)

# 모델에 따라 추가적인 preprocessing 필요한 경우 있음 

model = models.Sequential([
        # relu 쓰면, kernel_initializer='he_normal'
        # sigmoid, tanh 에는 default (glorot_normal)
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1)),  # Input 레이어 추가
        layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
model.summary()

# 모델 정의
def build_model(optimizer='adam', learning_rate=1e-3):
    build_model = model    
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    # 하이퍼파라미터 튜닝 설정
    
    build_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
    )
    
    return build_model


keras_clf = KerasClassifier(build_model, optimizer='adam', learning_rate=1e-3)

param_distribs = {
    "optimizer": ["adam", "rmsprop", "sgd"],
    "learning_rate": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}

# EarlyStopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Model Chekpoint (가장 좋은 모델 저장) 저장 안하는 게 좋을듯..? 사용하려면, callbacks에 추가
model_checkpoint = keras.callbacks.ModelCheckpoint(
    './models/????.keras',  # 모델 저장 경로, model 이름으로
    monitor='val_accuracy',
    save_best_only=True,  # 가장 좋은 모델만 저장
    mode='max',  # val_accuracy가 최대일 때 저장
    verbose=1
)



# 모델 학습
grid_search_cv = GridSearchCV(keras_clf, param_distribs, cv=StratifiedKFold(n_splits=5))
grid_search_cv.fit(X, y, epochs=EPOCHS, callbacks=[early_stopping], verbose=1)

best_model = grid_search_cv.best_estimator_

history = best_model.fit(
    X, y,  # X and y are your training data, no need to split again
    epochs=EPOCHS,
    validation_data=(X_val, y_val),  # Use the validation data
    verbose=1,
    callbacks=[early_stopping]
)

# 모델 예측 및 정확도 계산
y_true = y_val
y_pred = np.argmax(best_model.predict(X_val), axis=1) 

# accuracy_score 계산
final_accuracy = accuracy_score(y_true, y_pred)
print(f"Final training accuracy using sklearn's accuracy_score: {final_accuracy * 100:.2f}%")

# 학습 및 검증 정확도 그래프
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Top-2 Accuracy 그래프
plt.figure(figsize=(10, 6))
plt.plot(history.history['sparse_top_k_categorical_accuracy'], label='Training Top-2 Accuracy')
plt.plot(history.history['val_sparse_top_k_categorical_accuracy'], label='Validation Top-2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Top-2 Accuracy')
plt.title('Training and Validation Top-2 Accuracy')
plt.legend()
plt.grid(True)
plt.show()
