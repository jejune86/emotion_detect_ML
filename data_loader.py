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
classes = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]
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

    # 학습 데이터에는 증강 적용
    non_happy_indices = y_train != classes.index("happy")
    X_train_non_happy = X_train[non_happy_indices]
    y_train_non_happy = y_train[non_happy_indices]
    
    # Convert lists to np.array
    augmented_images = augment_image(X_train_non_happy)
    
    # augment된 데이터 추가
    X_train_augmented = np.concatenate([X_train, np.array(augmented_images)], axis=0)
    y_train_augmented = np.concatenate([y_train, np.array(y_train_non_happy)], axis=0)

    # Return the augmented train and validation data
    return all_images, all_labels, X_train_augmented, y_train_augmented, X_val, y_val