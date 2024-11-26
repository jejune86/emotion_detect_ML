import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_path = "./dataset/train"
test_path = "./dataset/test"
RANDOM_STATE = 486
VALIDATION_SIZE = 0.2
DEFAULT_SIZE = 48

# 클래스 레이블 정의
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

NUM_CLASSES = len(classes)



def load_train_data(
                     img_size=DEFAULT_SIZE,
                     gray=False,
                     normalization=True,
                     flatten=False,
                                                ):

    # 전체 데이터셋을 로드
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        color_mode="grayscale" if gray else "rgb",  # batch_size를 None으로 설정하여 전체 데이터셋을 한 번에 로드
        batch_size=32768,
        shuffle=False,
        seed=RANDOM_STATE,
        class_names=classes
    )

    # 한 번에 numpy 배열로 변환
    all_images = np.vstack([images.numpy() for images, _ in train_dataset])
    all_labels = np.hstack([labels.numpy() for _, labels in train_dataset])


    # Stratified split: 데이터를 훈련 세트와 검증 세트로 나누기
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, test_size=VALIDATION_SIZE, stratify=all_labels, random_state=RANDOM_STATE
    )

    def augment_image(images):
        # 이미지 증강을 위한 레이어 정의
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1)
        ])
        
        # 증강 적용
        augmented = augmentation_layers(images)
        return augmented

    # happy와 disgusted 클래스에 대한 처리를 분리
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


    def preprocess_image(image, label):
        if normalization:
            image = image / 255.0
        if flatten:
            image = image.flatten()
        return image, label
    
    
    X_train_augmented, y_train_augmented = preprocess_image(X_train_augmented, y_train_augmented)
    X_val, y_val = preprocess_image(X_val, y_val)


    # # disgusted 클래스 증강 결과 시각화
    # augmented_batch = augment_image(X_train_disgusted[:5])
    # visualize_augmented_images(X_train_disgusted[:5], augmented_batch)
    
    # # 다른 클래스 증강 결과 시각화
    # augmented_batch_other = augment_image(X_train_other[:5])
    # visualize_augmented_images(X_train_other[:5], augmented_batch_other)
    
    
    #TODO 추가적인 전처리 필요한 경우 여기에
    # ex) DenseNet169 
    # X_train_augmented = tf.keras.applications.densenet.preprocess_input(X_train_augmented)
    # X_val = tf.keras.applications.densenet.preprocess_input(X_val)
    
    
    # 증강된 데이터를 tf.data.Dataset으로 변환
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_augmented, y_train_augmented))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # 성능 최적화를 위한 데이터 파이프라인 설정
    train_ds = train_ds.shuffle(buffer_size=len(X_train_augmented))

    return train_ds, val_ds



def visualize_augmented_images(original_images, augmented_images, num_samples=5):
    plt.figure(figsize=(15, 3*num_samples))
    
    for idx in range(min(num_samples, len(original_images))):
        # 원본 이미지
        plt.subplot(num_samples, 2, 2*idx + 1)
        img = original_images[idx]
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img.astype(np.uint8), vmin=0, vmax=255)
        plt.title(f'Original Image {idx+1}')
        plt.axis('off')
        
        # Augmented image
        plt.subplot(num_samples, 2, 2*idx + 2)
        aug_img = augmented_images[idx]
        if isinstance(aug_img, tf.Tensor):
            aug_img = aug_img.numpy()
        if len(aug_img.shape) == 2:
            plt.imshow(aug_img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(aug_img.astype(np.uint8), vmin=0, vmax=255)
        plt.title(f'Augmented Image {idx+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
