import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

train_path = "./dataset/train"
test_path = "./dataset/test"
RANDOM_STATE = 486
VALIDATION_SIZE = 0.2
DEFAULT_SIZE = 48

# 클래스 레이블 정의
classes = ["angry", "disgusted", "happy", "neutral", "sad", "surprised"]
NUM_CLASSES = len(classes)

def load_train_data(
    img_size=DEFAULT_SIZE,
    gray=False,
    normalization=True,
    flatten=False,
    batch_size=64,
):


    # 라벨을 stratified하게 분리하기 위해 이미지와 라벨을 수동으로 추출
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        color_mode="grayscale" if gray else "RGB",
        batch_size=batch_size,
        shuffle=False,  # shuffle=False로 먼저 불러오고 stratified sampling 처리
        seed=RANDOM_STATE,
        class_names = classes
    )

    # 라벨과 이미지 데이터를 리스트로 가져오기
    all_images = []
    all_labels = []
    for image_batch, label_batch in train_dataset:
        all_images.append(image_batch.numpy())
        all_labels.append(label_batch.numpy())

    # 배치 처리된 데이터를 하나의 배열로 합침
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Stratified split: 데이터를 훈련 세트와 검증 세트로 나누기
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, test_size=VALIDATION_SIZE, stratify=all_labels, random_state=RANDOM_STATE
    )

    # TensorFlow Dataset으로 변환
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # 데이터 전처리 함수 정의
    def preprocess_image(image, label):
        if normalization:
            image = image / 255.0  # 정규화
        if flatten:
            image = tf.reshape(image, [-1])  # 평탄화
        return image, label

    def augment_image(image, label):
        if label != classes.index("happy") :
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label

    # 학습 데이터에는 증강 적용
    train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y))
    
    non_happy_dataset = train_dataset.filter(lambda x, y: y != classes.index("happy"))
    
    augmented_dataset = non_happy_dataset.map(lambda x, y: augment_image(x, y))
    
    train_dataset = train_dataset.concatenate(augmented_dataset).batch(batch_size = batch_size)

    # 검증 데이터에는 증강 미적용
    val_dataset = val_dataset.map(lambda x, y: preprocess_image(x, y)).batch(batch_size = batch_size)

    # 데이터 셔플 및 프리페치
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=RANDOM_STATE).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset