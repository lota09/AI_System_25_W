import os
import cv2
import mediapipe as mp
import csv
import numpy as np

# --- Configuration ---
DATASET_PATH = r"c:\Users\lota\Developments\AISys_25_W\term_project\assets\Synthetic ASL Numbers"
OUTPUT_TRAIN_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\train_data.csv"
OUTPUT_TEST_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\test_data.csv"

# Target labels (folders to process)
TARGET_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def process_folder(base_path, output_csv):
    print(f"Processing dataset: {base_path} -> {output_csv}")
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: label, x0, y0, z0, ..., x20, y20, z20
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
        writer.writerow(header)
        
        total_samples = 0
        success_samples = 0
        
        for label in TARGET_LABELS:
            folder_path = os.path.join(base_path, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found {folder_path}")
                continue
                
            print(f"  Processing Class '{label}'...")
            
            for filename in os.listdir(folder_path):
                if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.jpeg')):
                    continue
                    
                total_samples += 1
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue

                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Normalize landmarks relative to wrist (point 0)
                        # This makes the model robust to hand position in the frame
                        wrist = hand_landmarks.landmark[0]
                        base_x, base_y, base_z = wrist.x, wrist.y, wrist.z
                        
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            # Use relative coordinates for better generalization
                            norm_x = lm.x - base_x
                            norm_y = lm.y - base_y
                            norm_z = lm.z - base_z
                            landmarks.extend([norm_x, norm_y, norm_z])
                        
                        writer.writerow([label] + landmarks)
                        success_samples += 1
                        break # Only take the first detected hand per image

    print(f"Finished {base_path}. Extracted {success_samples}/{total_samples} samples ({success_samples/total_samples*100:.1f}%)")

if __name__ == "__main__":
    # Process Train Data
    process_folder(os.path.join(DATASET_PATH, "Train_Nums"), OUTPUT_TRAIN_CSV)
    
    # Process Test Data
    process_folder(os.path.join(DATASET_PATH, "Test_Nums"), OUTPUT_TEST_CSV)
    
    print("Done!")
