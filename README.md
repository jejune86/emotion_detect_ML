# emotion_detect_ML


### TODO
- 각각 맡은 모델 조사하기
    - 각 모델의 형태(model.summary(), 코드) 
    - layer가 어떤 구성으로 되어 있는지
    - 장점, 어디에 사용되는지
---
- 모델 넣고 학습시키기
    - 모델에 따라 input_size, gray/rgb, flatten 다름, 바꿔주기
    - 모델에 따라 hyperparameter tuning하기
        - dropout, neuron 수... (batch size는 후에 할 예정)
        - 시간, epoch 수 너무 커지지 않게 고려하기
---
- 결과 저장
    - 최종 accuracy
    - hyperparameter tune 결과
    - Learning Curve 2개
    - Confusion Matrix
 --- 
++ 최종 val accuracy는 50 +- 로 나와야 함, 결과가 20~30선 이라면 뭔가 잘못된 것


### 예정
- 결과로 받은 모델에서 ensemble 해볼 예정
- 모델 활용한 간단한 프로그램
- 보고서 작성
    - 위 3개 역할 분담해서 나눠서 할 예정