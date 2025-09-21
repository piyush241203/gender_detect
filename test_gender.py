import cv2
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("final_gender_classifier_model1.h5")

# Image size expected by model
IMAGE_SIZE = (224, 224)

# Labels
labels = ["Female", "Male"]

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face region using Haarcascade (optional enhancement)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess face
        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict gender
        prediction = model.predict(face_input)[0][0]
        predicted_class = 1 if prediction > 0.5 else 0
        label = f"{labels[predicted_class]} ({prediction:.2f})"

        # Draw rectangle and label
        color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame
    cv2.imshow("Gender Classification", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
