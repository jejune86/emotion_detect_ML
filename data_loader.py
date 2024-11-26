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
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
# disgusted는 데이터가 너무 부족하여, 사용 x

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
        image_size=(img_size, img_size),
        color_mode="grayscale" if gray else "rgb",
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
    
    # 데이터 전처리 함수 정의
    def preprocess_image(image, label):
        if normalization:
            image = image / 255.0  # 정규화
        if flatten:
            image = image.flatten()  # 평탄화
        return image, label

    all_images, all_labels = preprocess_image(all_images, all_labels)
    
    # Stratified split: 데이터를 훈련 세트와 검증 세트로 나누기
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, test_size=VALIDATION_SIZE, stratify=all_labels, random_state=RANDOM_STATE
    )


    def augment_image(images):

        images = np.array([np.fliplr(img) for img in images])
        
        brightness_factor = np.random.uniform(0.9, 1.1, size=images.shape[0])
        images = np.array([np.clip(img * factor, 0, 255).astype(np.uint8) for img, factor in zip(images, brightness_factor)])
        
        contrast_factor = np.random.uniform(0.8, 1.2, size=images.shape[0])
        images = np.array([np.clip(img * factor, 0, 255).astype(np.uint8) for img, factor in zip(images, contrast_factor)])

        return images

    # happy와 disgusted 클래스에 대한 처리를 분리
    non_happy_indices = y_train != classes.index("happy")
    disgusted_indices = y_train == classes.index("disgusted")
    other_indices = ~(disgusted_indices | (y_train == classes.index("happy")))

    # disgusted 클래스의 데이터
    X_train_disgusted = X_train[disgusted_indices]
    y_train_disgusted = y_train[disgusted_indices]
    
    # 나머지 클래스의 데이터 (happy 제외)
    X_train_other = X_train[other_indices]
    y_train_other = y_train[other_indices]

    # disgusted 클래스는 3배로 증강
    augmented_disgusted = np.concatenate([
        augment_image(X_train_disgusted),
        augment_image(X_train_disgusted),
        augment_image(X_train_disgusted)
    ])
    
    y_train_disgusted_aug = np.repeat(y_train_disgusted, 3)

    # 다른 클래스들은 1번만 증강
    augmented_other = augment_image(X_train_other)
    
    # 모든 데이터 합치기
    X_train_augmented = np.concatenate([
        X_train,
        augmented_disgusted,
        augmented_other
    ], axis=0)
    y_train_augmented = np.concatenate([
        y_train,
        y_train_disgusted_aug,
        y_train_other
    ], axis=0)

    # 증강된 데이터를 tf.data.Dataset으로 변환
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_augmented, y_train_augmented))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Dataset 설정 (섞기, 배치 처리)
    train_ds = train_ds.shuffle(buffer_size=len(X_train_augmented))
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds