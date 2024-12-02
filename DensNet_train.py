import os
from tqdm import tqdm
from PIL import Image
from loss_function import focal_loss_sparse
import numpy as np
import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_train_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE

# GPU 설정 (필요에 따라 수정하세요)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 불필요한 경고 메시지 숨기기

# 하이퍼파라미터 설정
INPUT_SIZE = DEFAULT_SIZE
EPOCHS = 10
BATCH_SIZE = 64

# 하이퍼파라미터 그리드
learning_rates = [1e-4, 1e-3, 1e-2]
optimizers = ['adam', 'sgd_momentum']
activation_functions = ['relu', 'elu']
growth_rates = [12, 24]
dropout_rates = [0.2, 0.3]

def dense_block(x, blocks, growth_rate, name, activation='relu', dropout_rate=0.2):
    for i in range(blocks):
        y = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        y = layers.Activation(activation, name=f'{name}_act1_{i}')(y)
        y = layers.Conv2D(4 * growth_rate, 1, use_bias=False, kernel_initializer='he_normal', name=f'{name}_conv1_{i}')(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.BatchNormalization(name=f'{name}_bn2_{i}')(y)
        y = layers.Activation(activation, name=f'{name}_act2_{i}')(y)
        y = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_conv2_{i}')(y)
        y = layers.Dropout(dropout_rate)(y)
        x = layers.Concatenate(axis=-1)([x, y])
    return x

def transition_block(x, reduction, name, activation='relu'):
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation(activation, name=f'{name}_act')(x)
    channels = int(x.shape[-1] * reduction)
    x = layers.Conv2D(channels, 1, use_bias=False, kernel_initializer='he_normal', name=f'{name}_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=f'{name}_pool')(x)
    return x

def build_densenet(input_shape=(48, 48, 1), growth_rate=12, blocks=[6, 12, 24, 16],
                   activation='relu', dropout_rate=0.2, num_classes=7):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation(activation, name='act1')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    for i, block_size in enumerate(blocks):
        x = dense_block(x, block_size, growth_rate, name=f'dense{i+1}', activation=activation, dropout_rate=dropout_rate)
        if i != len(blocks) - 1:
            x = transition_block(x, 0.5, name=f'trans{i+1}', activation=activation)
    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.Activation(activation, name='act_final')(x)
    x = layers.GlobalAveragePooling2D(name='pool_final')(x)
    x = layers.Dropout(dropout_rate, name='dropout_final')(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='predictions')(x)
    return Model(inputs, outputs, name='densenet')

# 데이터 로드
train_dataset, validation_dataset = load_train_data(img_size=INPUT_SIZE, gray=True, normalization=True, flatten=False)

best_val_accuracy = 0
best_model = None
best_history = None
best_params = None

# Grid Search
for lr in learning_rates:
    for opt_name in optimizers:
        for activation_name in activation_functions:
            for growth_rate in growth_rates:
                for dropout_rate in dropout_rates:
                    print(f"\nTrying parameters: lr={lr}, optimizer={opt_name}, "
                          f"activation={activation_name}, growth_rate={growth_rate}, "
                          f"dropout={dropout_rate}")
                    try:
                        # 배치 처리를 제거하고 prefetch만 적용
                        train_ds = train_dataset.prefetch(tf.data.AUTOTUNE)
                        val_ds = validation_dataset.prefetch(tf.data.AUTOTUNE)

                        # 모델 생성
                        model = build_densenet(
                            input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
                            growth_rate=growth_rate,
                            activation=activation_name,
                            dropout_rate=dropout_rate,
                            num_classes=NUM_CLASSES
                        )

                        # 옵티마이저 설정
                        if opt_name == 'adam':
                            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        else:
                            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

                        # 모델 컴파일
                        model.compile(
                            optimizer=optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
                        )

                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=5,
                            restore_best_weights=True
                        )

                        # 모델 학습
                        history = model.fit(
                            train_ds,
                            epochs=EPOCHS,
                            validation_data=val_ds,
                            callbacks=[early_stopping],
                            verbose=1
                        )

                        val_metrics = model.evaluate(val_ds, verbose=0)
                        val_accuracy = val_metrics[1]

                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_model = model
                            best_history = history.history
                            best_params = {
                                'learning_rate': lr,
                                'optimizer': opt_name,
                                'activation': activation_name,
                                'growth_rate': growth_rate,
                                'dropout_rate': dropout_rate
                            }

                        # 메모리 정리
                        tf.keras.backend.clear_session()

                    except Exception as e:
                        print(f"Error in training with parameters: lr={lr}, optimizer={opt_name}, "
                              f"activation={activation_name}, growth_rate={growth_rate}, dropout={dropout_rate}")
                        print(f"Error message: {str(e)}")
                        continue

print(f"\n최적 파라미터: {best_params}")
print(f"최고 검증 정확도: {best_val_accuracy:.4f}")

# 결과 시각화
if best_history is not None:
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

    # Confusion Matrix
    y_true = []
    y_pred = []

    for images, labels in validation_dataset:
        predictions = best_model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()