import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ----------------------------
# Load model safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "emotion_model.keras")

model = load_model(model_path)

# Emotion labels
emotion_labels = [
    'angry','disgust','fear','happy','neutral','sad','surprise'
]

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ----------------------------
# UI
# ----------------------------
st.title("🎭 Facial Emotion Recognition System")
st.write("Upload an image to detect emotion")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

# ----------------------------
# Main logic
# ----------------------------
if uploaded_file is not None:

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", channels="BGR")

    # Predict button
    if st.button("🎯 Predict Emotion"):

        # IMPORTANT: copy image (avoid overwriting original)
        output_image = image.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no face detected
        if len(faces) == 0:
            st.warning("No face detected in image!")
        else:
            for (x, y, w, h) in faces:

                # Crop face
                face = output_image[y:y+h, x:x+w]

                # Preprocess for CNN
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48,48))
                gray_face = gray_face / 255.0
                gray_face = np.reshape(gray_face, (1,48,48,1))

                # Predict emotion
                pred = model.predict(gray_face, verbose=0)
                label = emotion_labels[np.argmax(pred)]

                # Draw rectangle + label
                cv2.rectangle(output_image, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(output_image, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0,255,0), 2)

            # Show final result
            st.image(output_image, caption="Prediction Result", channels="BGR")