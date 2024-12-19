import cv2
import numpy as np
import dlib
from collections import deque
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Dlib’s face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Use deques for dynamic array handling
arr_left = deque(maxlen=20)  # Keeps the last 20 entries for smoothing
arr_right = deque(maxlen=20)
decision_left = deque(maxlen=20)
decision_right = deque(maxlen=20)

# Font for displaying data
font = cv2.FONT_HERSHEY_SIMPLEX

# Fixation Timer Variables
fixation_start = None
fixation_duration = 0
previous_gaze_point = None  # To track previous gaze point

# Threshold to detect significant gaze movement
gaze_shift_threshold = 10

# Predefined fixation rectangle area
fixation_rect = (200, 120, 240, 60)

def calculate_gaze_point(landmarks):
    """Calculate the average gaze point based on eye landmarks."""
    left_eye_center = np.mean([(landmarks.part(37).x, landmarks.part(37).y),
                               (landmarks.part(40).x, landmarks.part(40).y)], axis=0)
    right_eye_center = np.mean([(landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(46).x, landmarks.part(46).y)], axis=0)
    gaze_point = np.mean([left_eye_center, right_eye_center], axis=0)
    return gaze_point

def is_within_fixation_area(landmarks):
    """Check if eye landmarks are within the predefined fixation area."""
    return (fixation_rect[0] <= landmarks.part(37).x <= fixation_rect[0] + fixation_rect[2] and
            fixation_rect[1] <= landmarks.part(37).y <= fixation_rect[1] + fixation_rect[3] and
            fixation_rect[0] <= landmarks.part(46).x <= fixation_rect[0] + fixation_rect[2] and
            fixation_rect[1] <= landmarks.part(46).y <= fixation_rect[1] + fixation_rect[3])

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        if is_within_fixation_area(landmarks):
            current_gaze_point = calculate_gaze_point(landmarks)
            
            if previous_gaze_point is None:
                previous_gaze_point = current_gaze_point
                fixation_start = time.time()
                fixation_duration = 0
            else:
                distance = np.linalg.norm(current_gaze_point - previous_gaze_point)
                
                if distance > gaze_shift_threshold:
                    # Gaze moved significantly, reset fixation
                    fixation_start = time.time()
                    fixation_duration = 0
                    previous_gaze_point = current_gaze_point
                else:
                    # Continue current fixation
                    fixation_duration = (time.time() - fixation_start) * 1000  # Milliseconds
            
            # Draw eyes
            left_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], np.int32)
            right_eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], np.int32)
            cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)

            # Compute eye bounding boxes
            left_min_x, left_min_y = np.min(left_eye_region, axis=0)
            left_max_x, left_max_y = np.max(left_eye_region, axis=0)
            right_min_x, right_min_y = np.min(right_eye_region, axis=0)
            right_max_x, right_max_y = np.max(right_eye_region, axis=0)

            left_eye = gray[left_min_y:left_max_y, left_min_x:left_max_x]
            right_eye = gray[right_min_y:right_max_y, right_min_x:right_max_x]

            _, threshold_left = cv2.threshold(left_eye, 70, 255, cv2.THRESH_BINARY_INV)
            _, threshold_right = cv2.threshold(right_eye, 70, 255, cv2.THRESH_BINARY_INV)

            left_white = cv2.countNonZero(threshold_left[:, :threshold_left.shape[1] // 2])
            right_white = cv2.countNonZero(threshold_right[:, threshold_right.shape[1] // 2:])

            arr_left.append(left_white)
            arr_right.append(right_white)

            # Median and smoothing logic
            if len(arr_left) > 2 and len(arr_right) > 2:
                median_left = np.median(arr_left)
                median_right = np.median(arr_right)
                decision_left.append(median_left)
                decision_right.append(median_right)

                avg_left = np.mean(decision_left)
                avg_right = np.mean(decision_right)

                # Display left/right eye data
                cv2.putText(frame, f"Left: {int(avg_left)}", (30, 180), font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right: {int(avg_right)}", (30, 200), font, 0.7, (0, 255, 0), 2)

    # Draw fixation rectangle
    cv2.rectangle(frame, (fixation_rect[0], fixation_rect[1]), 
                  (fixation_rect[0] + fixation_rect[2], fixation_rect[1] + fixation_rect[3]), 
                  (0, 255, 0), 3)

    # Display fixation time
    cv2.putText(frame, f"Fixation Time: {fixation_duration / 1000:.2f} s", 
            (30, 50), font, 0.7, (255, 255, 0), 2)

    cv2.imshow("Gaze Detection with Fixation Time", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
