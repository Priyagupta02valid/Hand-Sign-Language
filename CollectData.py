import os
import cv2

# Initialize camera
cap = cv2.VideoCapture(0)
directory = r"D:\ML_Projects\Sign Language\Images"

# Ensure all alphabet folders exist
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

while True:
    _, frame = cap.read()

    # Define ROI (Region of Interest)
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    roi = frame[40:400, 0:300]  # Extract the region inside the box
    cv2.imshow("ROI", roi)

    # Wait for key press
    interrupt = cv2.waitKey(10)

    # Get the pressed key as a character
    key_pressed = chr(interrupt & 0xFF).upper()

    # Check if the pressed key is a valid alphabet
    if key_pressed in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        folder_path = os.path.join(directory, key_pressed)
        
        # Count existing images in the folder
        count = len(os.listdir(folder_path))
        
        # Save image with an incrementing filename
        file_path = os.path.join(folder_path, f"{count + 1}.png")
        cv2.imwrite(file_path, roi)

        print(f"Saved {file_path}")

# Release resources
cap.release()  
cv2.destroyAllWindows()
