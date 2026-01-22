# 프로젝트 계획: 딥러닝 기반 미국 수화(ASL) 숫자 인식 시스템

이 문서는 외부 데이터셋(`Synthetic ASL Numbers`)을 활용하여 강건한 ASL 숫자 인식 모델을 개발하기 위한 수정된 계획입니다.

## 1. 프로젝트 개요

*   **목표:** 다양한 사람의 손을 인식할 수 있는 강건한(Robust) 딥러닝 기반 ASL 숫자(1~10) 분류기 개발.
*   **전략:**
    *   **Feature Extraction:** Mediapipe Hands를 사용하여 이미지에서 손 랜드마크(21개 좌표)를 추출.
    *   **Classification:** 추출된 좌표를 입력으로 받는 DNN(Deep Neural Network)을 학습시켜 숫자 분류.
*   **데이터셋:** `assets/Synthetic ASL Numbers` (Train/Test 분리됨).

## 2. 상세 단계별 계획

### Phase 1: 데이터 전처리 (Feature Extraction)
**목표:** 이미지 데이터셋을 머신러닝 학습용 좌표 데이터(.csv)로 변환.

1.  **데이터셋 구조 분석:**
    *   `Train_Nums/`: 학습용 이미지 (약 3,600장)
    *   `Test_Nums/`: 검증용 이미지 (약 1,100장)
    *   각 폴더 내 파일명 규칙 확인 필요 (라벨링 파싱).

2.  **전처리 스크립트 리팩토링 (`preprocess_dataset.py`):**
    *   `mediapipe`를 사용하여 폴더 내 모든 이미지를 순차적으로 로드.
    *   각 이미지에서 손 랜드마크(x, y, z * 21개) 추출.
    *   **정규화(Normalization):** 손의 크기나 위치에 상관없이 인식되도록, 손목(0번 랜드마크)을 원점(0,0)으로 이동시키고 크기를 정규화하는 로직 추가.
    *   **저장:** `train_data.csv`, `test_data.csv` 형태로 저장 (컬럼: label, x1, y1, z1, ..., x21, y21, z21).
    *   *주의:* Mediapipe가 손을 감지하지 못한 이미지는 데이터셋에서 제외.

### Phase 2: 딥러닝 모델 학습 (Model Training)
**목표:** 좌표 패턴을 숫자로 매핑하는 DNN 모델 개발.

1.  **모델 설계 (`model.py`):**
    *   **Framework:** PyTorch
    *   **구조:** Input(63) -> FC(128) + ReLU -> FC(64) + ReLU -> Output(10, Softmax)
    *   가볍고 빠른 추론을 위해 심플한 MLP 구조 채택.

2.  **학습 스크립트 작성 (`train.py`):**
    *   `train_data.csv` 로드 (Custom Dataset/DataLoader).
    *   Loss Function: CrossEntropyLoss.
    *   Optimizer: Adam.
    *   모델 가중치 저장 (`best_model.pth`).

3.  **검증:**
    *   `test_data.csv`를 사용하여 정확도(Accuracy) 측정.

### Phase 3: 실시간 추론 및 시스템 통합
**목표:** 학습된 모델을 사용하여 비디오 파일 또는 웹캠에서 실시간 인식.

1.  **통합 스크립트 작성 (`inference.py` 또는 `hand_pose_final.py`):**
    *   기존 `hand_pose.py` (영상 처리) + 학습된 `best_model.pth` (추론) 결합.
    *   비디오 프레임 -> Mediapipe 좌표 추출 -> 전처리(정규화) -> PyTorch 모델 입력 -> 예측값(숫자) 출력.
    *   화면에 인식된 숫자와 확률(Confidence) 표시.

2.  **최종 테스트:**
    *   제공된 과제용 비디오(`asl_alphabet.mp4`)로 최종 성능 검증.
    *   손의 방향이나 조명 변화에 잘 대응하는지 확인.

## 3. 예상 일정

*   **1단계:** 데이터셋 전처리 및 CSV 변환 (이미지 -> 좌표) **[우선순위]**
*   **2단계:** PyTorch 모델 구현 및 학습.
*   **3단계:** 추론 코드 통합 및 비디오 테스트.
