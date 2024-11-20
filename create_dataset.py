import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

for dir_ in os.listdir(DATA_DIR):
    if dir_ == '.gitignore':
        continue
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []
        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        
        # Check for valid file extension
        if not any(img_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"Skipping non-image file: {img_full_path}")
            continue
        
        # Load the image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Could not read image: {img_full_path}")
            continue
        
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error processing image {img_full_path}: {e}")
            continue
        
        # Process with Mediapipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            data.append(data_aux)
            labels.append(dir_)

# Save to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset creation completed successfully!")
