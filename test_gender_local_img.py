import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained gender model
model = load_model("final_gender_classifier_model1.h5")

# Constants
IMAGE_SIZE = (224, 224)
labels = ["Female", "Male"]  # ⚠️ Make sure this matches your training dataset order

# Load OpenCV DNN face detector
# prototxt = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "deploy.prototxt.txt")
# weights = cv2.data.haarcascades.replace("haarcascade_frontalface_default.xml", "res10_300x300_ssd_iter_140000.caffemodel")
# Load OpenCV DNN face detector
prototxt = r"C:\Codings\Personal Open AI\OpenCV\Age_Gender\models\deploy.prototxt.txt"
weights = r"C:\Codings\Personal Open AI\OpenCV\Age_Gender\models\res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, weights)


net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# Path to local image
image_path = r"C:\Users\Hp\Downloads\sample2.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("❌ Error: Could not load image.")
    exit()

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence_face = detections[0, 0, i, 2]

    if confidence_face > 0.5:  # filter weak detections
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Preprocess
        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)  # ✅ Match training preprocessing
        face_normalized = face_rgb / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict gender
        prediction = model.predict(face_input)[0][0]  # sigmoid output
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else 1 - prediction
        label = f"{labels[predicted_class]} ({confidence*100:.1f}%)"

        # Draw rectangle & label
        color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

# Show result
cv2.imshow("Gender Prediction - Local Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
