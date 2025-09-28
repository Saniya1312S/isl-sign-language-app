# This script performs real-time sign language alphabet prediction using a webcam,
# with continuous text-to-speech feedback.

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import threading
import os 
import pickle

# --- Step 1: Initialize Mediapipe and TTS Engine ---

# Initialize the hand landmarks model ONCE for efficiency
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize the text-to-speech engine
engine = pyttsx3.init()
# Set a slightly faster speech rate to reduce clipping
current_rate = engine.getProperty('rate')
engine.setProperty('rate', current_rate + 50) 
engine.setProperty('volume', 1.0)

# --- Global Audio State Variables (Used for continuous audio logic) ---
speech_in_progress = False
COOLDOWN_TIME = 0.25  # Time (in seconds) between repeated speech of the same sign

# Initialize shared state variables OUTSIDE the loop
last_speech_time = time.time()
last_spoken_sign = None 
current_predicted_sign = "No Sign Detected"
# Dedicated thread variable for speaking
speech_thread = None

def speak_sign(sign_text):
    """
    Handles the TTS process: speaks the text, waits for it to finish, and safely 
    stops the engine. This function runs in a separate thread.
    """
    global speech_in_progress
    if not speech_in_progress and sign_text != "No Sign Detected":
        speech_in_progress = True
        try:
            # We don't use print here, as it can slow down the thread
            engine.say(sign_text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            # This catches common driver/resource errors
            print(f"Internal Audio Error: {e}")
        finally:
            speech_in_progress = False

# --- Step 2: Load the Trained Model and Label Encoder ---

try:
    # NOTE: Using user-specified file names for the alphabet model
    model = joblib.load("alphabet_model.pkl") 
    # Load the label encoder
    le = joblib.load("alphabet_label_encoder.pkl")
    print("Alphabet Model and label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Alphabet model or label encoder files not found.")
    print("Please ensure 'alphabet_model.pkl' and 'alphabet_label_encoder.pkl' are in the directory.")
    exit()

# === Main Webcam Loop ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("\n--- Starting Real-time Alphabet Prediction ---")
print("Press 'q' to quit.")

while cap.isOpened():
    # Variables initialized outside the loop are directly accessible (no 'global' needed here)

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_predicted_sign = "No Sign Detected"
    features = np.zeros(126) # Initialize the feature vector

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            # Extract keypoints
            keypoints = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            label = handness.classification[0].label
            
            # Assign to left/right features (mirror effect correction)
            if label == "Left":
                features[63:] = keypoints  # Our Right Hand
            else:
                features[:63] = keypoints  # Our Left Hand
        
        # Predict the sign only if at least one hand was detected
        if np.sum(features) != 0:
            pred_idx = model.predict(features.reshape(1, -1))[0]
            current_predicted_sign = le.inverse_transform([int(pred_idx)])[0]

    # --- AUDIO FEEDBACK (CONTINUOUS LOGIC) ---
    current_time = time.time()
    
    # Check if a new sign is predicted OR if the cooldown has expired
    sign_changed = (current_predicted_sign != last_spoken_sign) and (current_predicted_sign != "No Sign Detected")
    cooldown_expired = (current_time - last_speech_time) > COOLDOWN_TIME

    # Only speak if: 1. A new sign is predicted OR 2. Cooldown is expired AND a sign is held
    # AND 3. The dedicated speech thread is NOT currently running a command
    if (sign_changed or (cooldown_expired and current_predicted_sign != "No Sign Detected")) and not speech_in_progress:
        
        # Update time and last spoken sign
        last_speech_time = current_time 
        last_spoken_sign = current_predicted_sign
        
        # Start the thread safely
        speech_thread = threading.Thread(target=speak_sign, args=(current_predicted_sign,), daemon=True)
        speech_thread.start()

    # Display the predicted sign on the screen
    cv2.putText(frame, f"Prediction: {current_predicted_sign}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Real-Time Alphabet Sign Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
