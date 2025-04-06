import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to your dataset folder
DATA_PATH = "D:/ML_Projects/Sign Language/Images"

# Define classes
actions = sorted(os.listdir(DATA_PATH))  # A-Z folders
num_classes = len(actions)

# Extract landmarks/keypoints or use raw image resizing
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Skipping unreadable image: {image_path}")
        return None
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.flatten() / 255.0

X, y = [], []

for label, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    files = os.listdir(action_path)[:20]  # limit to 20 images
    for file in files:
        img_path = os.path.join(action_path, file)
        img = process_image(img_path)
        if img is None:
            continue  # skip unreadable image
        X.append(img)
        y.append(label)

X = np.array(X)
y = to_categorical(y, num_classes=num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic feedforward model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Model trained and saved")
