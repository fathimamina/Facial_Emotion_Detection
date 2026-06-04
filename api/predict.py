import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load model (IMPORTANT: must be outside function for speed)
model = load_model("models/emotion_model.keras")

labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    file = request.files["file"]
    image = Image.open(file)

    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray / 255.0
    gray = np.reshape(gray, (1, 48, 48, 1))

    pred = model.predict(gray)
    label = labels[np.argmax(pred)]

    return {
        "statusCode": 200,
        "body": json.dumps({"emotion": label})
    }