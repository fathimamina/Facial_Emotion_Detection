import cv2
import numpy as np
from ultralytics import YOLO 
from tensorflow.keras.models import load_model


yolo = YOLO('yolov8n.pt')

#Load emotion model
emotion_model = load_model('models/emotion_model.keras')

#Emotion_labels
emotion_labels = [
        'angry','disgust','fear','happy','neutral','sad','surprise'
]

#start webcam
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    
    #yolo detection
    results = yolo(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1,y1,x2,y2 = map(int,box)

            #crop detected region
            face = frame[y1:y2,x1:x2]

            if face.size==0:
                continue 
            # Preprocess for emotion model
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48,48))
            gray = gray / 255.0
            gray = np.reshape(gray, (1,48,48,1))

            # Predict emotion
            pred = emotion_model.predict(gray, verbose=0)
            label = emotion_labels[np.argmax(pred)]

            # Draw bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Put label
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

    # Show frame
    cv2.imshow("FER System", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

