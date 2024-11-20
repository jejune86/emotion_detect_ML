import tensorflow as tf
from PIL import Image

train_path = "./dataset/train"
test_path = "./dataset/test"
RANDOM_STATE = 486
VALIDATION_SIZE = 0.2
DEFAULT_SIZE = 48

# 클래스 레이블 정의
labels = ["angry", "happy", "neutral", "sad", "surprised"]
NUM_CLASSES = len(labels)

def load_data(
    type,
    img_size=DEFAULT_SIZE,
    gray=False,
    normalization=True,
    flatten=False,
    augment=False,
    batch_size=64,
):
    if type == "train":
        path = train_path
    elif type == "test":
        path = test_path
    else:
        raise ValueError("Invalid dataset type. Choose 'train' or 'test'.")

    # 학습 및 검증 데이터셋 로딩
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        color_mode = "rgb",
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_STATE,
        validation_split=VALIDATION_SIZE,
        subset="training",
        class_names=labels,
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        color_mode = "rgb",
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_STATE,
        validation_split=VALIDATION_SIZE,
        subset="validation",
        class_names=labels,
    )

    # 데이터 전처리 함수 정의
    def preprocess_image(image, label, augment=False):
        # 그레이스케일 변환
        if gray:
            image = tf.image.rgb_to_grayscale(image)  # 이미지를 그레이스케일로 변환

        # 정규화
        if normalization:
            image = image / 255.0  # 픽셀 값을 0~1로 정규화

        # 데이터 증강 (학습 데이터만 적용)
        if augment:
            image = tf.image.random_flip_left_right(image)  
            image = tf.image.random_brightness(image, max_delta=0.1) 
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 

        # 이미지 평탄화 (옵션)
        if flatten:
            image = tf.reshape(image, [-1])  # 평탄화

        return image, label

    # 학습 데이터에는 증강 적용
    train_dataset = train_dataset.map(lambda x, y: preprocess_image(x, y, augment=augment))
    # 검증 데이터에는 증강 미적용
    val_dataset = val_dataset.map(lambda x, y: preprocess_image(x, y))

    return train_dataset, val_dataset
