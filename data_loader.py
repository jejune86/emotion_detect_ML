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

def load_train_data(img_size=DEFAULT_SIZE, gray=False, normalization=True, flatten=False):
   # 전체 데이터셋을 로드
   train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
       train_path,
       labels="inferred", 
       label_mode="int",
       image_size=(img_size, img_size),
       color_mode="grayscale" if gray else "rgb",
       batch_size=28709,
       shuffle=False,
       seed=RANDOM_STATE,
       class_names=classes
   )

   # numpy 배열로 변환
   images, labels = next(iter(train_dataset))
   all_images = images.numpy()
   all_labels = labels.numpy()
   print("Original image shape before preprocessing:", all_images.shape)  # 여기에 추가


   # Stratified split
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
   print("Shape after augmentation:", X_train_augmented.shape)  # 여기에 추가
    
   
   y_train_augmented = np.concatenate([
       y_train,
       y_train_disgusted_aug,
       y_train_other
   ], axis=0)
   

   def preprocess_image(image, label):
       print("Input image shape:", image.shape)  # 디버깅을 위한 출력
       
       # 입력 이미지가 (64, 48, 48) 형태인 경우 처리
       if len(image.shape) == 3:
           image = tf.reshape(image, (48, 48, 1))
       # 입력 이미지가 (64, 64, 48, 48) 형태인 경우 처리
       elif len(image.shape) == 4:
           image = tf.reshape(image, (48, 48, 1))
       
       if normalization:
           image = tf.cast(image, tf.float32) / 255.0
           
       print("Output image shape:", image.shape)  # 디버깅을 위한 출력
       return image, label

   # X_train_augmented shape 확인
   print("X_train_augmented shape:", X_train_augmented.shape)
   print("y_train_augmented shape:", y_train_augmented.shape)

   # 증강된 데이터를 tf.data.Dataset으로 변환
   train_ds = tf.data.Dataset.from_tensor_slices(
       (X_train_augmented, y_train_augmented)
   ).map(preprocess_image).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

   val_ds = tf.data.Dataset.from_tensor_slices(
       (X_val, y_val)
   ).map(preprocess_image).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

   # 최종 shape 확인
   for images, labels in train_ds.take(1):
       print("Final batch shape:", images.shape)
       print("Labels shape:", labels.shape)

       print("Final shape before creating dataset:", X_train_augmented.shape)
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