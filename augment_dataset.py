import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# Configuration
INPUT_ROOT = (
    r"c:\Users\lota\Developments\AISys_25_W\term_project\assets\ASL_Digits_masked"
)
OUTPUT_ROOT = (
    r"c:\Users\lota\Developments\AISys_25_W\term_project\assets\AUG_ASL_Digits_masked"
)
TARGET_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
AUGMENT_COUNT = 14  # Generate 3 augmented versions per image (Total 4x data)
ROTATION_RANGE = (-75, 75)  # Degrees


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Fill background with black or grey?
    # Dataset seems simply lit. Let's use black (0,0,0).
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
    return rotated


def augment_dataset(subset_name):
    input_path = os.path.join(INPUT_ROOT, subset_name)
    output_path = os.path.join(OUTPUT_ROOT, subset_name)

    print(f"Augmenting {subset_name}...")

    for label in TARGET_LABELS:
        input_folder = os.path.join(input_path, label)
        output_folder = os.path.join(output_path, label)

        if not os.path.exists(input_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)

        files = [
            f
            for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        for filename in tqdm(files, desc=f"Class {label}"):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Save original
            base_name, ext = os.path.splitext(filename)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_orig{ext}"), image)

            # Save rotated versions
            for i in range(AUGMENT_COUNT):
                angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
                rotated_img = rotate_image(image, angle)
                cv2.imwrite(
                    os.path.join(output_folder, f"{base_name}_aug{i}{ext}"), rotated_img
                )


if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input path not found: {INPUT_ROOT}")
    else:
        augment_dataset("Train_Nums")
        augment_dataset("Test_Nums")
        print("Augmentation Complete.")
