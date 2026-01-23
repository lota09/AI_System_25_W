# ASL 숫자 손동작 인식 시스템 (ASL Number Recognition System)

본 프로젝트는 딥러닝(PyTorch)과 MediaPipe를 활용하여 미국 수화(ASL) 숫자 1~10을 실시간으로 인식하는 시스템입니다.

## 📂 파일 구성
*   `augment_dataset.py`: 데이터 증강 스크립트 (회전 변환 -75~+75도)
*   `preprocess_dataset.py`: 이미지에서 랜드마크 추출 및 CSV 변환 (멀티프로세싱 적용)
*   `capture_weak_classes.py`: 취약 클래스(4, 5, 10 등) 실시간 수집 및 라벨링 도구
*   `train_model.py`: PyTorch MLP 모델 학습 스크립트 (Oversampling 적용)
*   `hand_pose_final.py`: **최종 실행 파일** (실시간 웹캠/비디오 추론)
*   `assets/`: 결과 영상 (`final_result.mp4`) 및 데이터셋 폴더
*   `models/`: 학습된 모델 파일 (`best_model.pth`)

## 🚀 실행 방법 (How to Run)

### 1. 환경 설정 (Dependencies)
Python 3.10 환경을 권장합니다. MediaPipe와 최신 NumPy(2.0 이상) 간 충돌 방지를 위해 `numpy<2` 설정이 중요합니다.

```bash
pip install "numpy<2" torch torchvision mediapipe opencv-python pandas tqdm
```

또는

```bash
pip install -r requirement.txt
```

### 2. 최종 시스템 실행 (Inference)
이미 학습된 모델(`models/best_model.pth`)을 사용하여 바로 실행할 수 있습니다.
```bash
python hand_pose_final.py
```
*   실행 후 `1`을 선택하면 **실시간 웹캠**, `2`를 선택하면 **비디오 파일**을 입력으로 받습니다.
*   종료하려면 `q` 키를 누르세요.

---

## 🛠️ (선택) 처음부터 다시 학습하기 (Training from Scratch)
데이터셋이 준비되어 있다는 가정하에 아래 순서대로 실행합니다.

**Step 1: 데이터 증강**
```bash
python augment_dataset.py
```

**Step 2: 전처리 (랜드마크 추출)**
```bash
python preprocess_dataset.py
```

**Step 3: 취약 데이터 수집 (필요 시)**
```bash
python capture_weak_classes.py
```
*   웹캠 앞에서 4, 5, 10번 제스처를 취하고 해당 숫자 키를 눌러 데이터를 저장합니다.

**Step 4: 모델 학습**
```bash
python train_model.py
```
*   `train_data.csv`와 `train_data_weak.csv`를 합쳐 모델을 학습하고 `models/best_model.pth`로 저장합니다.

---