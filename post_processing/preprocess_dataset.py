import os
import cv2
import mediapipe as mp
import csv
import numpy as np

# --- Configuration ---
DATASET_PATH = (
    r"c:\Users\lota\Developments\AISys_25_W\term_project\assets\Synthetic ASL Numbers"
)
OUTPUT_TRAIN_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\train_data.csv"
OUTPUT_TEST_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\test_data.csv"

# Target labels (folders to process)
TARGET_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)


# --- Alignment Function (Method 1) ---
def align_landmarks(landmarks):
    """
    Aligns hand landmarks by:
    1. Translating wrist (0) to origin (0,0,0)
    2. Rotating so vector 0->9 (Middle Finger MCP) aligns with Y-axis
    3. Ignoring Z-rotation (assumes palm mostly facing camera) for simplcity
    4. Normalizing size based on distance 0->9
    """
    # 1. Convert to numpy array for easier math
    # Expected landmarks format: list of [x, y, z] relative to image (or world)
    # But mediapipe gives normalized [0,1]. We can use them directly.
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # 2. Translation: Center wrist at (0,0,0)
    wrist = coords[0]
    coords -= wrist

    # 3. Rotation: Align 0->9 vector to Y-axis
    # Vector 0->9
    v_0_9 = coords[9]

    # Calculate angle in XY plane
    theta = np.arctan2(v_0_9[0], v_0_9[1])  # Angle with Y-axis?
    # Actually arctan2(y, x) is normal angle.
    # We want to rotate so (x,y) becomes (0, y_new) i.e. x=0.
    # Current angle of vector from Y-axis (positive Y is down in image coords? Wait. Mediapipe Y is down.)
    # Let's align to negative Y axis (up) which is standard "upright" hand?
    # Or align to positive Y (down)?
    # Let's align to Y-axis (x=0).
    # Angle of vector with standard Y-axis (0,1).

    # Rotation matrix around Z-axis
    rotation_angle = -theta
    # However arctan2(y,x) gives angle from X-axis.
    # We want vector to lie on Y axis.
    # Angle from X-axis should become 90 deg (pi/2) or -90 deg (-pi/2).
    # Easier: Just construct a 2D rotation.

    # Let's compute rotation angle to make X component 0 and Y component positive (or negative).
    # v = [x, y]. Target = [0, |v|] (or [0, -|v|])
    # Let's align 0->9 to point UP (-Y in image coords)

    target_angle = -np.pi / 2  # -90 degrees (pointing up)
    current_angle = np.arctan2(v_0_9[1], v_0_9[0])
    rotation_angle = target_angle - current_angle

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array(((c, -s), (s, c)))

    # Apply rotation to X, Y (keep Z same)
    coords[:, :2] = np.dot(coords[:, :2], R.T)

    # 4. Scaling: Normalize size based on length of 0->9
    # (After rotation, 0->9 is aligned to Y axis, so length is just abs(y) of point 9... roughly)
    # Better use actual distance.
    dist_0_9 = np.linalg.norm(
        v_0_9
    )  # Original distance (it's invariant under rotation)
    scale = 1.0 / dist_0_9
    coords *= scale

    # Return flattened list
    return coords.flatten().tolist()


def process_folder(base_path, output_csv):
    print(f"Processing dataset: {base_path} -> {output_csv}")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header: label, x0, y0, z0, ..., x20, y20, z20
        header = ["label"] + [
            f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
        ]
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
                if not (
                    filename.lower().endswith(".jpg")
                    or filename.lower().endswith(".png")
                    or filename.lower().endswith(".jpeg")
                ):
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
                        # Use our new alignment function
                        aligned_landmarks = align_landmarks(hand_landmarks.landmark)

                        writer.writerow([label] + aligned_landmarks)
                        success_samples += 1
                        break  # Only take the first detected hand per image

    print(
        f"Finished {base_path}. Extracted {success_samples}/{total_samples} samples ({success_samples / total_samples * 100:.1f}%)"
    )


if __name__ == "__main__":
    # Process Train Data
    process_folder(os.path.join(DATASET_PATH, "Train_Nums"), OUTPUT_TRAIN_CSV)

    # Process Test Data
    process_folder(os.path.join(DATASET_PATH, "Test_Nums"), OUTPUT_TEST_CSV)

    print("Done!")
