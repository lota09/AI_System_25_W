# 딥러닝 기반 ASL 숫자 손동작 인식 시스템 프로젝트 보고서

## 1. 프로젝트 개요 (Overview)
*   **목표**: 딥러닝 기술을 활용하여 미국 수화(ASL)의 숫자 0~9(또는 1~10) 손동작을 실시간으로 정확히 인식하고 분류하는 시스템을 구현합니다.
*   **핵심 과제**: 
    1.  **데이터 부족 및 불균형 해결**: 제한된 외부 데이터셋을 효과적으로 활용하여 학습 성능을 극대화해야 합니다.
    2.  **환경 강건성(Robustness) 확보**: 카메라 각도, 손의 회전, 크기 변화 등 다양한 실제 입력 환경에서도 안정적인 인식률을 달성해야 합니다.
*   **개발 환경**: 
    *   **Language**: Python 3.10
    *   **Frameworks**: PyTorch (Model Training), MediaPipe (Feature Extraction), OpenCV (Image Processing)
    *   **Platform**: Windows 11 (CUDA GPU Acceleration)

## 2. 시스템 구조 (Architecture)
본 프로젝트는 **하이브리드 파이프라인(Hybrid Pipeline)** 방식을 채택하였습니다. 시스템은 크게 **특징 추출기(Feature Extractor)**와 **분류 신경망(Classification Network)**으로 나뉩니다.

### 2.1 Feature Extractor: Google MediaPipe Hands
*   **Input**: RGB Video Frame (640x480)
*   **Process**: 영상 내에서 손을 실시간으로 추적하여 **21개의 3D 관절 랜드마크(Keypoints)**를 추출합니다.
*   **Output**: 21개 랜드마크 × (x, y, z) 좌표 = **63차원 벡터**

### 2.2 Classification Network: PyTorch MLP (Multi-Layer Perceptron)
추출된 63차원 좌표 벡터를 입력받아 최종 숫자를 분류하는 **완전 연결 신경망(Fully Connected Network)**입니다. 경량화와 실시간성을 최우선으로 고려하여 설계되었습니다.

*   **Model Type**: Feed-Forward Neural Network (MLP)
*   **Total Layers**: **3 Fully Connected Layers** (Input -> Hidden -> Hidden -> Output)
*   **Layer Details**:
    1.  **Input Layer**:
        *   Input Size: 63 (Features)
        *   Output Size: 128 (Neurons)
        *   Activation: ReLU
        *   Regularization: Dropout (p=0.2)
    2.  **Hidden Layer**:
        *   Input Size: 128
        *   Output Size: 64 (Neurons)
        *   Activation: ReLU
        *   Regularization: Dropout (p=0.2)
    3.  **Output Layer**:
        *   Input Size: 64
        *   Output Size: 10 (Classes: Digit 1~10)
        *   Activation: Softmax (Implicit in CrossEntropyLoss)

*   **Parameter Count**: 약 17,000개 수준의 초경량 모델로, CPU 환경에서도 1ms 미만의 추론 속도를 보장합니다.

## 3. 데이터 처리 및 학습 전략 (Data Strategy)
본 프로젝트의 핵심은 **"데이터 중심의 문제 해결(Data-Centric AI)"** 접근 방식입니다. 초기 제공된 데이터셋이 전무하였기에, 외부 데이터셋을 능동적으로 분석 및 가공하여 문제를 해결하였습니다.

### 3.1 데이터셋 구성 (Datasets)
Kaggle에서 특성이 상이한 두 종류의 데이터셋을 선정하여 하이브리드 구성을 취했습니다.
1.  **Dataset A**: [Synthetic ASL Numbers](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers) (11,000장, 3D 합성)
    *   **활용**: 증강 없이 Baseline 학습용으로 사용.
2.  **Dataset B**: [American Sign Language Digits](https://www.kaggle.com/datasets/victoranthony/asl-digits-0-9) (700장, 실사, 검은 배경)
    *   **활용**: **15배 증강**을 위한 시드(Seed) 데이터로 사용.

### 3.2 강건성 확보를 위한 3단계 전략 (3-Step Robustness Strategy)
회전과 각도 변화에 취약했던 초기 모델의 한계를 극복하기 위해 다음의 3단계 전략을 수립하였습니다.

**Step 1: 데이터 증강 (Data Augmentation)**
*   `Dataset B`에 대해 **-75도 ~ +75도** 회전을 적용, 15배로 증강하여 다양한 손의 기울기를 학습시켰습니다.

**Step 2: 기하학적 정렬 (Canonical Pose Alignment)**
*   학습 및 추론 단계에서 입력 손 좌표를 기하학적으로 정규화하여 모델의 난이도를 낮췄습니다.
    *   **Translation**: 손목(0번)을 (0,0) 원점으로 이동.
    *   **Rotation**: 손목(0번) -> 중지(9번) 벡터가 **Y축(수직)**을 향하도록 회전 변환.
*   **효과**: 모델은 회전된 손도 항상 "똑바로 선 정자세"로 인식하게 됩니다.

**Step 3: 취약 클래스(Weak Class) 집중 보강 (Active Learning)**
*   **Human-in-the-loop**: `4`, `5`, `10`번 오분류를 해결하기 위해 사용자가 직접 실시간 수집 도구(`capture_weak_classes.py`)로 약 700개의 약점 데이터를 캡처했습니다.
*   **Oversampling**: 해당 데이터를 학습 시 **4배 오버샘플링**하여 취약 패턴을 집중 학습시켰습니다.

## 4. 실험 결과 (Experimental Results)
*   **학습 데이터**: 약 88,000개 (증강 포함) + 2,800개 (약점 데이터 오버샘플링 적용)
*   **검증 정확도(Validation Accuracy)**: **99.85%** 달성
*   **실제 테스트 성능**: 
    *   웹캠 실시간 테스트에서 손을 360도 가까이 회전시켜도 안정적으로 인식함을 확인.
    *   초기에 구분이 어려웠던 4, 5, 10번 숫자의 오분류가 완벽하게 개선됨.

## 5. 결론
본 프로젝트는 **데이터 증강, 기하학적 정렬(Alignment), 능동적 오차 분석(Active Learning)**을 유기적으로 결합하여, 제한된 데이터 환경에서도 상용 수준의 강건함을 갖춘 ASL 인식 시스템을 성공적으로 구현하였습니다.
