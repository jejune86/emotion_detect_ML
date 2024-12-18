import tensorflow as tf
import numpy as np
from data_loader import load_train_data, load_test_data, NUM_CLASSES, DEFAULT_SIZE, classes
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gc
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, 
    f1_score, roc_curve, auc, roc_auc_score
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 설정

# 하이퍼파라미터 설정
INPUT_SIZE = DEFAULT_SIZE
EPOCHS = 10
BATCH_SIZE = 32

# 데이터 로드
#train_ds, val_ds = load_train_data(img_size=INPUT_SIZE, gray=False, batch_size=BATCH_SIZE)
test_ds = load_test_data(img_size=INPUT_SIZE, gray=False, batch_size=BATCH_SIZE)
# 개별 모델 로드
dense_model = tf.keras.models.load_model('./models/DenseNet.keras')
vgg_model = tf.keras.models.load_model('./models/VGG.keras')
res_model = tf.keras.models.load_model('./models/ResNet.keras')

# Soft Voting 앙상블 함수
def soft_voting_ensemble(models, dataset):
    y_true = []
    y_pred_probs = []

    for images, labels in dataset:
        # 각 모델의 예측 확률 수집
        predictions = [model.predict(images, verbose=0) for model in models]
        
        # 평균 확률 계산 (Soft Voting)
        avg_predictions = np.mean(predictions, axis=0)
        y_pred_probs.extend(avg_predictions)
        
        # 실제 레이블 수집
        y_true.extend(labels.numpy())

    # 최종 예측값: 확률에서 argmax로 클래스 결정
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_true, y_pred, np.array(y_pred_probs)

# Soft Voting 실행
models = [dense_model, vgg_model, res_model]
y_true, y_pred, y_pred_probs  = soft_voting_ensemble(models, test_ds)

# Confusion Matrix 출력
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Soft Voting Ensemble')
plt.show()

# 최종 검증 정확도 출력
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"앙상블 검증 정확도: {accuracy:.4f}")

# 정확도, Precision, Recall, F1 Score 계산
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"앙상블 검증 정확도: {np.mean(np.array(y_true) == np.array(y_pred)):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# AUC 및 ROC Curve 계산 (다중 클래스)
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
auc_scores = []
plt.figure(figsize=(10, 8))

for i, class_name in enumerate(classes):
    # 각 클래스에 대한 ROC Curve
    fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.4f})')

# 전체 AUC 평균 출력
mean_auc = np.mean(auc_scores)
print(f"Mean AUC: {mean_auc:.4f}")

# ROC Curve 시각화
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Soft Voting Ensemble')
plt.legend(loc="lower right")
plt.show()