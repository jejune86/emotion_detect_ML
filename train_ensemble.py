import tensorflow as tf
import numpy as np
from data_loader import load_train_data, NUM_CLASSES, DEFAULT_SIZE
from ResNet_train import build_resnet
from train_VGG import build_vgg
from train_DensNet import build_densenet
from loss_function import focal_loss_sparse
import os

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 하이퍼파라미터 설정
INPUT_SIZE = DEFAULT_SIZE
EPOCHS = 10
BATCH_SIZE = 64

def train_ensemble():
    # 데이터 로드
    train_dataset, validation_dataset = load_train_data(
        img_size=INPUT_SIZE, 
        gray=True, 
        normalization=True, 
        flatten=False
    )

    # 데이터셋 배치 설정
    train_ds = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ResNet 모델 생성 및 학습
    resnet = build_resnet(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
        num_classes=NUM_CLASSES,
        activation='elu'
    )
    resnet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss_sparse(),
        metrics=['accuracy']
    )
    resnet.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

    # DenseNet 모델 생성 및 학습
    densenet = build_densenet(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
        growth_rate=12,
        activation='relu',
        dropout_rate=0.2,
        num_classes=NUM_CLASSES
    )
    densenet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss_sparse(),
        metrics=['accuracy']
    )
    densenet.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

    # VGG 모델 생성 및 학습
    vgg = build_vgg(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
        num_classes=NUM_CLASSES,
        activation='relu'
    )
    vgg.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss_sparse(),
        metrics=['accuracy']
    )
    vgg.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

    # Soft Voting을 위한 앙상블 예측
    def ensemble_predict(models, dataset):
        predictions = []
        for model in models:
            pred = model.predict(dataset)
            predictions.append(pred)
        
        # 평균 계산 (Soft Voting)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    # 검증 데이터에 대한 앙상블 예측
    models = [resnet, densenet, vgg]
    ensemble_predictions = ensemble_predict(models, val_ds)
    
    # 앙상블 모델의 정확도 계산
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    y_pred = np.argmax(ensemble_predictions, axis=1)
    ensemble_accuracy = np.mean(y_true == y_pred)
    
    print(f"\n앙상블 모델 검증 정확도: {ensemble_accuracy:.4f}")

if __name__ == "__main__":
    train_ensemble()