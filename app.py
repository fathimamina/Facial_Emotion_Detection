import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("models/emotion_model.keras")

labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

st.title("🎭 Facial Emotion Detection App")

st.write("Upload a face image to detect emotion")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray / 255.0
    gray = gray.reshape(1, 48, 48, 1)

    # Predict
    pred = model.predict(gray, verbose=0)
    label = labels[np.argmax(pred)]

    st.subheader(f"Predicted Emotion: {label} 😄")