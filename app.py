# app.py

from function import extract_keypoints
from keras.models import model_from_json
import cv2
import numpy as np
import os

# Load model architecture + weights
with open("model.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("model.weights.h5")

# Classes (A–Z)
DATA_PATH = "D:/ML_Projects/Sign Language/Images"
actions = sorted(os.listdir(DATA_PATH))
threshold = 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI box
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    crop = frame[40:400, 0:300]

    # Extract features from this single frame
    keypoints = extract_keypoints(crop)          # 1D array of length 4096
    input_data = keypoints.reshape(1, -1)        # shape (1, 4096)

    # Predict
    res = model.predict(input_data)[0]
    idx = np.argmax(res)
    predicted_class = actions[idx]
    confidence = res[idx]

    # Display result
    label = f"{predicted_class} ({confidence*100:.1f}%)" if confidence > threshold else "-"
    cv2.rectangle(frame, (0, 0), (350, 40), (245, 117, 16), -1)
    cv2.putText(frame, f"Output: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
