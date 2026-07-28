import cv2
import os
import numpy as np

# -----------------------------
# Create Face Recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

dataset_path = "dataset"

# -----------------------------
# Check Dataset Folder
# -----------------------------
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    input("\nPress Enter to close...")
    exit()

# -----------------------------
# Read All Student Images
# -----------------------------
for student_id in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_id)

    if not os.path.isdir(student_folder):
        continue

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        ids.append(int(student_id))

# -----------------------------
# Check Images
# -----------------------------
if len(faces) == 0:
    print("No face images found!")
    input("\nPress Enter to close...")
    exit()

print("--------------------------------")
print("Training Started...")
print("--------------------------------")

# -----------------------------
# Train Model
# -----------------------------
recognizer.train(faces, np.array(ids))

# -----------------------------
# Create Trainer Folder
# -----------------------------
os.makedirs("trainer", exist_ok=True)

# -----------------------------
# Save Model
# -----------------------------
recognizer.save("trainer/trainer.yml")

print()
print("======================================")
print("   MODEL TRAINING COMPLETED")
print("======================================")
print(f"Total Students : {len(set(ids))}")
print(f"Total Images   : {len(faces)}")
print("Model Saved    : trainer/trainer.yml")
print("======================================")

# -----------------------------
# Keep Window Open
# -----------------------------
input("\nPress Enter to close...")