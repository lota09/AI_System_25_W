import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
MODEL_PATH = r".\models\best_model.pth"
OUTPUT_VIDEO = r".\assets\final_result.mp4"


# --- Model Definition ---
class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.layer1 = nn.Linear(63, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)  # 10 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# --- Alignment Function (Method 1: Rotation + Translation + Scale) ---
def align_landmarks(landmarks):
    """
    Aligns hand landmarks by:
    1. Translating wrist (0) to origin (0,0,0)
    2. ROTATING so vector 0->9 (Middle Finger MCP) aligns with negative Y-axis (Up)
    3. Normalizing size based on distance 0->9
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # 1. Translation
    wrist = coords[0]
    coords -= wrist

    # 2. Rotation: Align 0->9 vector to Y-axis
    v_0_9 = coords[9]

    # Calculate angle to rotate (target: -90 degrees / pointing up)
    target_angle = -np.pi / 2
    current_angle = np.arctan2(v_0_9[1], v_0_9[0])  # Angle from X-axis
    rotation_angle = target_angle - current_angle

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array(((c, -s), (s, c)))

    # Apply rotation to X, Y
    coords[:, :2] = np.dot(coords[:, :2], R.T)

    # 3. Scaling
    dist_0_9 = np.linalg.norm(coords[9])  # Re-calc after rotation
    if dist_0_9 > 0:
        scale = 1.0 / dist_0_9
        coords *= scale

    return coords.flatten().tolist()


# --- Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandGestureModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

model.eval()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Input Selection
print("========================================")
print(" Select Input Source:")
print(" 1. Real-time Camera")
print(" 2. Video File")
print("========================================")
choice = input("Enter choice (1/2): ").strip()

cap = None
source_fps = 30.0
is_video_file = False

if choice == "1":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)
elif choice == "2":
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not file_path:
        sys.exit(1)
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        sys.exit(1)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    is_video_file = True
else:
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, source_fps, (width, height))

frame_count = 0
label_map = {i: str(i + 1) for i in range(10)}
target_frame_duration = 1.0 / source_fps

print("Processing started...")

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    ann_frame = frame.copy()

    predicted_label = "Unknown"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                ann_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Use Method 2 Alignment
            aligned_features = align_landmarks(hand_landmarks.landmark)

            input_tensor = torch.tensor([aligned_features], dtype=torch.float32).to(
                device
            )
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                predicted_label = label_map[pred_idx.item()]
                confidence = conf.item()

            with open("result.txt", "a") as f:
                f.write(f"Frame {frame_count}: {predicted_label} ({confidence:.4f})\n")

            text = f"Digit: {predicted_label} ({confidence * 100:.1f}%)"
            cv2.putText(
                ann_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            print(f"Frame {frame_count}: {text}")

    out.write(ann_frame)
    cv2.imshow("ASL Digit Recognition", ann_frame)

    processing_time = time.time() - start_time
    wait_time_ms = int(max(1, (target_frame_duration - processing_time) * 1000))
    if cv2.waitKey(wait_time_ms) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
