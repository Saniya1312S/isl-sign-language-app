import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm
import random

# ---- Configuration ----
DATA_DIRS = [
    r'D:\A_Sign\sign_cod\Data\alphabet\ISL_Dataset_2',
    r'D:\A_Sign\sign_cod\Data\alphabet\Indian'
]
SAVE_PATH = "alphabet_landmarks_2hands.pkl"
N_TRAIN = 200
N_TEST = 50
IMG_EXT = ('.jpg', '.jpeg', '.png')

mp_hands = mp.solutions.hands

def extract_2hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.45) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        left = [0.0]*63
        right = [0.0]*63
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
                if handedness.classification[0].label == 'Left':
                    left = arr.tolist()
                else:
                    right = arr.tolist()
        return left, right

# ---- Gather image paths by class ----
class_img_dict = {}
for ds in DATA_DIRS:
    for cls in sorted(os.listdir(ds)):
        path = os.path.join(ds, cls)
        if not os.path.isdir(path): continue
        files = []
        for f in os.listdir(path):
            if f.lower().endswith(IMG_EXT):
                files.append(os.path.join(path, f))
        if files:
            class_img_dict.setdefault(cls, []).extend(files)

# ---- Sampling and Shuffle ----
X_train, y_train, X_test, y_test = [], [], [], []
for cls, files in class_img_dict.items():
    files = list(set(files))  # remove dups
    random.shuffle(files)
    train_files = files[:N_TRAIN]
    test_files = files[N_TRAIN:N_TRAIN+N_TEST]

    # Process train images
    for imgp in tqdm(train_files, desc=f"{cls} (train)"):
        img = cv2.imread(imgp)
        if img is not None:
            left, right = extract_2hand_landmarks(img)
            X_train.append(left + right)
            y_train.append(cls)
    # Process test images
    for imgp in tqdm(test_files, desc=f"{cls} (test)"):
        img = cv2.imread(imgp)
        if img is not None:
            left, right = extract_2hand_landmarks(img)
            X_test.append(left + right)
            y_test.append(cls)

# ---- Save as dictionary ----
with open(SAVE_PATH, 'wb') as f:
    pickle.dump({
        'X_train': np.array(X_train, dtype=np.float32),
        'y_train': np.array(y_train),
        'X_test': np.array(X_test, dtype=np.float32),
        'y_test': np.array(y_test)
    }, f)
print(f"Saved {len(y_train)} train, {len(y_test)} test samples to {SAVE_PATH}")
