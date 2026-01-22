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


# --- Initialization ---
# 1. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandGestureModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

model.eval()

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# 3. Input Selection Logic
print("========================================")
print(" Select Input Source:")
print(" 1. Real-time Camera")
print(" 2. Video File")
print("========================================")
choice = input("Enter choice (1/2): ").strip()

cap = None
source_fps = 30.0  # Default fallback
is_video_file = False

if choice == "1":
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default camera.")
        sys.exit(1)
    print("Camera connected.")

elif choice == "2":
    print("Opening file dialog...")
    # Initialize basic Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring to front

    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(1)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    is_video_file = True
    print(f"Using Video input: {file_path} @ {source_fps:.2f} FPS")

else:
    print("Invalid choice. Exiting.")
    sys.exit(1)


# 4. Output Video Writer Setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, source_fps, (width, height))

print(f"Processing started. Output will be saved to {OUTPUT_VIDEO}")
print("Press 'q' to stop.")

# --- Main Loop ---
frame_count = 0
label_map = {i: str(i + 1) for i in range(10)}

# Calculate target frame duration in seconds
target_frame_duration = 1.0 / source_fps

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        if is_video_file:
            print("End of video file.")
        else:
            print("Failed to grab frame.")
        break

    frame_count += 1

    # Process Frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)

    image_rgb.flags.writeable = True
    ann_frame = frame.copy()

    predicted_label = "Unknown"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                ann_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract and Normalize Landmarks for Model
            wrist = hand_landmarks.landmark[0]
            base_x, base_y, base_z = wrist.x, wrist.y, wrist.z

            landmarks = []
            for lm in hand_landmarks.landmark:
                norm_x = lm.x - base_x
                norm_y = lm.y - base_y
                norm_z = lm.z - base_z
                landmarks.extend([norm_x, norm_y, norm_z])

            # Prediction
            input_tensor = torch.tensor([landmarks], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)

                predicted_label = label_map[pred_idx.item()]
                confidence = conf.item()

            # Save result to text file
            with open("result.txt", "a") as f:
                f.write(f"Frame {frame_count}: {predicted_label} ({confidence:.4f})\n")

            # Display Prediction on Frame
            text = f"Digit: {predicted_label} ({confidence * 100:.1f}%)"
            cv2.putText(
                ann_frame,
                text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Print to console
            print(f"Frame {frame_count}: {text}")

    # Write frame
    out.write(ann_frame)

    # Display Result
    cv2.imshow("ASL Digit Recognition", ann_frame)

    # FPS Locking Logic
    processing_time = time.time() - start_time
    wait_time_sec = target_frame_duration - processing_time
    wait_time_ms = int(max(1, wait_time_sec * 1000))

    if cv2.waitKey(wait_time_ms) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Result saved to {OUTPUT_VIDEO}")
