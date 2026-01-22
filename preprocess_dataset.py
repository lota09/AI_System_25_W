import os
import cv2
import mediapipe as mp
import csv
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count

# --- Configuration ---
# Use the NEW augmented dataset path
DATASET_PATH = (
    r"c:\Users\lota\Developments\AISys_25_W\term_project\assets\Merged_AUG_ASL"
)
OUTPUT_TRAIN_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\train_data.csv"
OUTPUT_TEST_CSV = r"c:\Users\lota\Developments\AISys_25_W\term_project\test_data.csv"

# Target labels (folders to process)
TARGET_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


# --- Alignment Function (Method 2: NO Rotation, Only Translation & Scale) ---
def align_landmarks(landmarks):
    """
    Aligns hand landmarks by:
    1. Translating wrist (0) to origin (0,0,0)
    2. Scaling based on distance 0->9 (Middle Finger MCP)
    3. NO ROTATION (Deep learning model will learn invariance)
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # 1. Translation: Center wrist at (0,0,0)
    wrist = coords[0]
    coords -= wrist

    # 2. Scaling: Normalize size based on length of 0->9
    v_0_9 = coords[9]
    dist_0_9 = np.linalg.norm(v_0_9)
    if dist_0_9 > 0:
        scale = 1.0 / dist_0_9
        coords *= scale

    # Return flattened list
    return coords.flatten().tolist()


def process_file_batch(file_paths, label):
    """
    Process a batch of files.
    Each thread/process needs its own MediaPipe instance because accuracy is better
    in 'static_image_mode' if we don't share instances across threads for processing completely different images (or just for safety).
    Actually, MediaPipe Hands IS thread-safe but initializing one per worker is cleaner.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    results_data = []

    for img_path in file_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                aligned_landmarks = align_landmarks(hand_landmarks.landmark)
                results_data.append([label] + aligned_landmarks)
                break

    hands.close()
    return results_data


def process_folder(base_path, output_csv):
    print(f"Processing dataset: {base_path} -> {output_csv}")

    all_data = []
    total_files_count = 0

    # Collect all file paths first
    tasks = []

    for label in TARGET_LABELS:
        folder_path = os.path.join(base_path, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found {folder_path}")
            continue

        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not files:
            continue

        total_files_count += len(files)

        # Split files into chunks for parallel processing
        # Chunk size depends on memory/CPU.
        # Making it too small adds overhead. Too big might be uneven.
        # Let's try dividing files into N chunks where N = CPU cores * 2
        num_workers = min(32, (os.cpu_count() or 4) * 2)
        chunk_size = max(1, len(files) // num_workers)

        for i in range(0, len(files), chunk_size):
            chunk = files[i : i + chunk_size]
            tasks.append((chunk, label))

    print(
        f"  Total files to process: {total_files_count}. Using parallel processing..."
    )

    # Use ProcessPoolExecutor for CPU-bound tasks (MediaPipe image processing)
    # However, MediaPipe releases GIL often, so ThreadPoolExecutor might also work and has less overhead.
    # But ProcessPool is safer for true parallelism in Python.
    # Let's stick with ProcessPoolExecutor.
    success_count = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map tasks to executor
        futures = [
            executor.submit(process_file_batch, chunk, label) for chunk, label in tasks
        ]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                all_data.extend(batch_results)
                success_count += len(batch_results)
                # print(f"    Batch finished. Total success so far: {success_count}", end='\r')
            except Exception as e:
                print(f"    Worker failed with error: {e}")

    print(f"\n  Writing results to {output_csv}...")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["label"] + [
            f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
        ]
        writer.writerow(header)
        writer.writerows(all_data)

    print(
        f"Finished {base_path}. Extracted {success_count}/{total_files_count} samples ({(success_count / total_files_count * 100) if total_files_count > 0 else 0:.1f}%)"
    )


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path not found: {DATASET_PATH}")
    else:
        # Process Train Data
        process_folder(os.path.join(DATASET_PATH, "Train_Nums"), OUTPUT_TRAIN_CSV)

        # Process Test Data
        process_folder(os.path.join(DATASET_PATH, "Test_Nums"), OUTPUT_TEST_CSV)

        print("Done!")
