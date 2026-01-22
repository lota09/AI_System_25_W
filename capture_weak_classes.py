import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import csv

# --- Configuration ---
MODEL_PATH = r".\models\best_model.pth"
OUTPUT_ADDITIONAL_CSV = r".\train_data_weak.csv"  # Save weak class data here


# --- Model Definition (Must match training) ---
class HandGestureModel(nn.Module):
    def __init__(self):
        super(HandGestureModel, self).__init__()
        self.layer1 = nn.Linear(63, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# --- Alignment Function (Method 1: Rotation + Translation + Scale) ---
def align_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords -= wrist

    v_0_9 = coords[9]
    target_angle = -np.pi / 2
    current_angle = np.arctan2(v_0_9[1], v_0_9[0])
    rotation_angle = target_angle - current_angle

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array(((c, -s), (s, c)))

    coords[:, :2] = np.dot(coords[:, :2], R.T)

    dist_0_9 = np.linalg.norm(coords[9])
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

# Initialize CSV if not exists
if not os.path.exists(OUTPUT_ADDITIONAL_CSV):
    with open(OUTPUT_ADDITIONAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [
            f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
        ]
        writer.writerow(header)

print("========================================")
print(" Interactive Weak Class Capture Tool")
print("========================================")
print(" 1. Run inference real-time.")
print(" 2. If model predicts WRONGLY (or low confidence),")
print("    press '0' ~ '9' key to save the CURRENT frame as correct label.")
print("    (e.g., press '4' to save current pose as label 4)")
print(" 3. Data will be appended to 'train_data_weak.csv'")
print(" 4. Press 'q' to quit.")
print("========================================")

cap = cv2.VideoCapture(0)
label_map = {i: str(i + 1) for i in range(10)}

saved_count = 0
last_saved_label = None
last_saved_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    predicted_label = "?"
    confidence = 0.0
    current_aligned_features = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features for prediction AND saving
            current_aligned_features = align_landmarks(hand_landmarks.landmark)

            # Predict
            input_tensor = torch.tensor(
                [current_aligned_features], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                predicted_label = label_map[pred_idx.item()]
                confidence = conf.item()

            # Display Prediction
            color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
            cv2.putText(
                frame,
                f"Pred: {predicted_label} ({confidence * 100:.1f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            break  # Only process first hand for capture tool

    # UI Feedback
    cv2.putText(
        frame,
        "Press 0-9 to save as GT label",
        (10, height := frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if time.time() - last_saved_time < 1.0:
        cv2.putText(
            frame,
            f"Saved as {last_saved_label}!",
            (width := frame.shape[1] - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.imshow("Weak Class Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif ord("0") <= key <= ord("9") and current_aligned_features is not None:
        # Key '1' -> label "1", Key '0' -> label "10"
        if key == ord("0"):
            target_label = "10"
        else:
            target_label = str(key - ord("0"))

        with open(OUTPUT_ADDITIONAL_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([target_label] + current_aligned_features)

        saved_count += 1
        last_saved_label = target_label
        last_saved_time = time.time()
        print(f"[{saved_count}] Saved current pose as Label: {target_label}")

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {saved_count} new samples to {OUTPUT_ADDITIONAL_CSV}")
