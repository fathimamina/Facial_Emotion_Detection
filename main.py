import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion model
emotion_model = load_model('models/emotion_model.keras')

emotion_labels = [
    'angry','disgust','fear','happy','neutral','sad','surprise'
]

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Preprocess
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48,48))
        gray = gray / 255.0
        gray = np.reshape(gray, (1,48,48,1))

        # Predict
        pred = emotion_model.predict(gray, verbose=0)
        label = emotion_labels[np.argmax(pred)]

        # Draw
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("FER System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()