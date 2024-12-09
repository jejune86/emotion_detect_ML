from tqdm import tqdm
import os
from PIL import Image
from loss_function import focal_loss_sparse  
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_train_data
from data_loader import NUM_CLASSES, DEFAULT_SIZE
from tensorflow.keras.regularizers import l2
import gc

#최적 파라미터: {'learning_rate': 0.0001, 'optimizer': 'adam', 'activation': 'elu'}
#최고 검증 정확도: 0.5808

gc.collect()

#tf.debugging.set_log_device_placement(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 로깅 레벨 조정 (WARNING 이상만 표시)

# GPU 메모리 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 메모리 제한 설정 (예: 4GB)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

print("TensorFlow has access to the following devices:", tf.config.list_physical_devices())

# See TensorFlow version
print("TensorFlow version:", tf.__version__)


# 하이퍼파라미터 설정
INPUT_SIZE = DEFAULT_SIZE
EPOCHS = 10
BATCH_SIZE = 64

# 하이퍼파라미터 그리드
learning_rates = [1e-4, 1e-3, 1e-2]
optimizers = ['adam', 'sgd_momentum']
activation_functions = ['relu', 'elu']

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, activation='relu'):
    shortcut = x

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                                 name=f'{name}_conv_shortcut' if name else None)(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_bn_shortcut' if name else None)(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                     kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                     kernel_regularizer=l2(1e-4),
                     name=f'{name}_conv1' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn1' if name else None)(x)

    if activation == 'relu':
        x = layers.ReLU(name=f'{name}_act1' if name else None)(x)
    else:
        x = layers.ELU(name=f'{name}_act1' if name else None)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                     kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                     kernel_regularizer=l2(1e-4),
                     name=f'{name}_conv2' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn2' if name else None)(x)

    x = layers.Add(name=f'{name}_add' if name else None)([shortcut, x])

    if activation == 'relu':
        x = layers.ReLU(name=f'{name}_act2' if name else None)(x)
    else:
        x = layers.ELU(name=f'{name}_act2' if name else None)(x)

    return x

def build_resnet(input_shape=(48, 48, 1), num_classes=7, activation='relu'):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Conv2D(64, 7, strides=2, padding='same',
                     kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                     kernel_regularizer=l2(1e-4),
                     name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)

    if activation == 'relu':
        x = layers.ReLU(name='act1')(x)
    else:
        x = layers.ELU(name='act1')(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    x = residual_block(x, 64, name='block2a', activation=activation)
    x = residual_block(x, 64, name='block2b', activation=activation)

    x = residual_block(x, 128, stride=2, name='block3a', activation=activation)
    x = residual_block(x, 128, name='block3b', activation=activation)

    x = residual_block(x, 256, stride=2, name='block4a', activation=activation)
    x = residual_block(x, 256, name='block4b', activation=activation)

    x = residual_block(x, 512, stride=2, name='block5a', activation=activation)
    x = residual_block(x, 512, name='block5b', activation=activation)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax',
                    kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
                    name='predictions')(x)

    return Model(inputs, x, name='resnet_emotion')

# 데이터 로드
train_dataset, validation_dataset = load_train_data(img_size=INPUT_SIZE, gray=True, normalization=True, flatten=False)

# 데이터 캐싱 및 프리페칭 (배치 제거)
train_ds = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)

# 디버깅 출력: 데이터 배치 형상 확인
for x_batch, y_batch in train_ds.take(1):
    print(f"배치 이미지 형태: {x_batch.shape}")
    print(f"배치 레이블 형태: {y_batch.shape}")

best_val_accuracy = 0
best_model = None
best_history = None
best_params = None

# Grid Search
for lr in learning_rates:
    for opt_name in optimizers:
        for activation_name in activation_functions:
            try:
                print(f"\nTrying parameters: lr={lr}, optimizer={opt_name}, activation={activation_name}")

                # 모델 생성
                model = build_resnet(
                    input_shape=(48, 48, 1),
                    num_classes=NUM_CLASSES,
                    activation=activation_name
                )

                # optimizer 설정
                if opt_name == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                elif opt_name == 'sgd_momentum':
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

                # 모델 컴파일
                model.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
                )

                # 조기 종료 콜백
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_sparse_categorical_accuracy',
                    patience=5,
                    restore_best_weights=True
                )

                # 학습
                history = model.fit(
                    train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[early_stopping],
                    verbose=1,
                    shuffle=True
                )

                # 검증 정확도 평가
                val_metrics = model.evaluate(val_ds)
                val_accuracy = val_metrics[1]

                print(f"Validation accuracy: {val_accuracy}")

                if best_val_accuracy is None or val_accuracy > best_val_accuracy:
                    if best_model is not None:
                        del best_model
                    best_val_accuracy = val_accuracy
                    best_model = model
                    best_history = history.history.copy()
                    best_params = {
                        'learning_rate': lr,
                        'optimizer': opt_name,
                        'activation': activation_name
                    }
                else:
                    del model

                # 메모리 정리
                tf.keras.backend.clear_session()

            except Exception as e:
                print(f"Error in training with parameters: {lr}, {opt_name}, {activation_name}")
                print(f"Error message: {str(e)}")
                continue

# 결과 확인
if best_history is None:
    print("No successful training runs. Please check the model and data.")
else:
    print(f"\n최적 파라미터: {best_params}")
    print(f"최고 검증 정확도: {best_val_accuracy:.4f}")
    best_model.summary()

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(best_history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(best_history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
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

    for images, labels in val_ds.take(-1):
        predictions = best_model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()