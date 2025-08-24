import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from pygame import mixer

# Initialize Pygame mixer for alarm
mixer.init()
mixer.music.load("alarm.mp3")

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Function to calculate EAR
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indexes from Mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20
frame_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye coordinates
            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # Draw eye contours
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            # Compute EAR
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CLOSED_FRAMES:
                    cv2.putText(frame, "WAKE UP....WAKE UP!", (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                    if not mixer.music.get_busy():
                        mixer.music.play(-1)  # loop alarm
            else:
                frame_counter = 0
                mixer.music.stop()

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
