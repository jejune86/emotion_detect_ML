from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models

# 데이터 로딩 함수
def load_data_and_labels(data_dir):
    X = []  # 이미지 데이터
    y = []  # 라벨

    # 라벨별로 데이터 로드
    for label in tqdm(os.listdir(data_dir), desc="Loading Labels"):
        label_path = os.path.join(data_dir, label)

        if os.path.isdir(label_path):  # 폴더만 처리
            for img_file in tqdm(os.listdir(label_path), desc=f"Loading Images for {label}"):
                img_path = os.path.join(label_path, img_file)
                try:
                    # 이미지를 로드하고 배열로 변환
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # 데이터와 라벨 추가
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # 이미지 데이터를 (n_samples, height * width * channels) 형태로 펼침
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)  # 2D 배열로 변환 (n_samples, n_features)

    # 라벨을 숫자로 변환
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

# 데이터 로드 및 전처리
data_dir = "./dataset/train"  # 데이터 경로
X, y = load_data_and_labels(data_dir)

# 데이터를 섞기
X, y = shuffle(X, y, random_state=42)

# 훈련/검증 데이터 나누기
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')  # Replace 'num_classes' with the actual number of classes
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습 중 진행도 표시 및 모델 훈련
epochs = 30
batch_size = 64

# 훈련하기
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)

# 최종 검증 정확도 출력
val_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {val_accuracy * 100:.2f}%")

# 모델 예측 및 정확도 계산
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)  # 확률을 클래스 인덱스로 변환
final_accuracy = accuracy_score(y_val, y_pred)
print(f"Final validation accuracy using sklearn's accuracy_score: {final_accuracy * 100:.2f}%")