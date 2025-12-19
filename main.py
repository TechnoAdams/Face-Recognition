import cv2
import numpy as np
import random
import time

# --- Setup ---
HAAR_CASCADE_FILE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

last_change_time = time.time()
current_emotion = random.choice(EMOTION_LABELS)
emotion_color = (0, 255, 0)

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(" Random Emotion Detector started. Press 'q' to quit.")
nice
# --- Helper function ---
def get_emotion_color(emotion):
    color_map = {
        'Happy': (0, 255, 0),
        'Sad': (255, 0, 0),
        'Surprise': (0, 255, 255),
        'Angry': (0, 0, 255),
        'Fear': (128, 0, 128),
        'Disgust': (0, 128, 0),
        'Neutral': (200, 200, 200)
    }
    return color_map.get(emotion, (255, 255, 255))

prev_time = time.time()
fps = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    # Calculate FPS
    now_time = time.time()
    fps = 1 / (now_time - prev_time)
    prev_time = now_time

    # Change emotion every 2 seconds
    if now_time - last_change_time > 2:
        current_emotion = random.choice(EMOTION_LABELS)
        emotion_color = get_emotion_color(current_emotion)
        last_change_time = now_time

    # Add a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # Draw info bar
    cv2.putText(frame, f"Emotion: {current_emotion}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, emotion_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw faces with soft UI elements
    for (x, y, w, h) in faces:
        # Face circle outline
        center = (x + w // 2, y + h // 2)
        radius = int((w + h) / 4)
        cv2.circle(frame, center, radius, emotion_color, 3, cv2.LINE_AA)

        # Emotion bubble above face
        label_bg = (x, y - 45)
        label_size = cv2.getTextSize(current_emotion, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (label_bg[0] - 10, label_bg[1] - 30),
                      (label_bg[0] + label_size[0] + 10, label_bg[1] + 10),
                      (*emotion_color, ), -1)
        cv2.putText(frame, current_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("🧠 Emotion Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()