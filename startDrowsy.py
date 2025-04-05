from scipy.spatial import distance
from imutils import face_utils
import pygame
import time
import dlib
import cv2
import os
from sendGmail import sendAlertEmail as sendAlertEmail
from threading import Thread
import Constant as Constants

def detectDrowsiness():
    # Initialize Pygame and load sound
    volume = 0.10
    people_info = {}
    dataset_path = 'datasets'
    
    # Inform user about face recognition initialization
    print('Recognizing Faces, Ensure Sufficient Lighting...')

    # Create lists for images, labels, names, and ids
    (face_images, labels, names, identifier) = ([], [], {}, 0)

    pygame.mixer.init()
    pygame.mixer.music.load('audio/alert.wav')
    pygame.mixer.music.stop()  # Ensure sound is stopped at start

    # Set the thresholds
    EYE_ASPECT_RATIO_THRESHOLD = Constants.threashold
    MOUTH_ASPECT_RATIO_THRESHOLD = 0.6
    
    # Set the minimum consecutive frames for detection
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
    YAWN_CONSEC_FRAMES = 30

    # Initialize counters
    frame_counter = 0
    warning_counter = 0
    yawn_counter = 0
    alert_active = False  # Flag to track if alert is currently playing

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def calculate_eye_aspect_ratio(eye_landmarks):
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2 * C)
        return ear

    def calculate_mouth_aspect_ratio(mouth_landmarks):
        # Vertical distances
        A = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        B = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        
        # Horizontal distance
        C = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        # Calculate MAR
        mar = (A + B) / (2.0 * C)
        return mar

    # Load face detector and predictor
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("dataset\shape_predictor_68_face_landmarks.dat")

    # Extract indexes of facial landmarks
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

    # Start webcam video capture
    video_capture = cv2.VideoCapture(Constants.source)
    
    # Get frame dimensions for positioning text
    ret, frame = video_capture.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
    else:
        frame_height, frame_width = 480, 640  # Default values

    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            break
            
        current_frame = cv2.flip(current_frame, 1)
        grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector(grayscale_frame, 0)
        face_rectangles = face_cascade.detectMultiScale(grayscale_frame, 1.3, 5)

        # If no faces are detected, reset counters and stop alert
        if len(detected_faces) == 0:
            frame_counter = 0
            warning_counter = 0
            yawn_counter = 0
            if alert_active:
                pygame.mixer.music.stop()
                alert_active = False
            

        for face in detected_faces:
            shape = shape_predictor(grayscale_frame, face)
            shape = face_utils.shape_to_np(shape)

            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(current_frame, (x, y), 2, (0, 255, 0), -1)

            # Get eye landmarks and calculate EAR
            left_eye_landmarks = shape[left_eye_start:left_eye_end]
            right_eye_landmarks = shape[right_eye_start:right_eye_end]
            left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye_landmarks)
            right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye_landmarks)
            eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

            # Get mouth landmarks and calculate MAR
            mouth_landmarks = shape[mouth_start:mouth_end]
            mouth_aspect_ratio = calculate_mouth_aspect_ratio(mouth_landmarks)

            # Draw contours
            left_eye_hull = cv2.convexHull(left_eye_landmarks)
            right_eye_hull = cv2.convexHull(right_eye_landmarks)
            mouth_hull = cv2.convexHull(mouth_landmarks)
            
            cv2.drawContours(current_frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(current_frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(current_frame, [mouth_hull], -1, (0, 255, 0), 1)

            # Draw face rectangles
            for (x, y, w, h) in face_rectangles:
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 128, 255), 2)

            Constants.driveCar = True

            # Drowsiness Detection
            if eye_aspect_ratio < EYE_ASPECT_RATIO_THRESHOLD:
                frame_counter += 1
                warning_counter += 1

                if not alert_active:
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play(-1)
                    alert_active = True
                    
                time.sleep(0.1)
                volume = min(volume + 0.01, 1.0)  # Cap volume at 1.0

                cv2.putText(current_frame, "Drowsiness Detected", (150, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(current_frame, "Eyes Closed", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
                Constants.driveCar = False

                if warning_counter > Constants.limit:
                    print('\nLimit Exceeded! Sending alert message to family members...')
                    pygame.mixer.music.stop()
                    alert_active = False
                    try:
                        Thread(target=sendAlertEmail).start()
                    except Exception as e:
                        print('Error: Message Not Sent')
                    warning_counter = 0
            else:
                cv2.putText(current_frame, "Eyes Open", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                if alert_active:
                    pygame.mixer.music.stop()
                    alert_active = False
                frame_counter = 0
                warning_counter = 0
                volume = 0.10

            # Yawning Detection
            if mouth_aspect_ratio > MOUTH_ASPECT_RATIO_THRESHOLD:
                yawn_counter += 1
                cv2.putText(current_frame, "Yawning Detected", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    cv2.putText(current_frame, "Excessive Yawning!", (150, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    try:
                        Thread(target=sendAlertEmail).start()
                    except Exception as e:
                        print('Error: Message Not Sent')
                    yawn_counter = 0
            else:
                yawn_counter = 0

            # Display MAR and EAR values on the right side of the screen
            # Calculate positions based on frame width
            text_x = frame_width - 200  # Adjust this value to move text left/right
            
            # Draw background rectangles for better visibility
            cv2.rectangle(current_frame, (text_x - 10, 20), (text_x + 150, 90), (0, 0, 0), -1)
            
            # Display values
            cv2.putText(current_frame, f"EAR: {eye_aspect_ratio:.2f}", (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(current_frame, f"MAR: {mouth_aspect_ratio:.2f}", (text_x, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show video feed
        cv2.imshow('Video Feed', current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            video_capture.release()
            cv2.destroyAllWindows()
            pygame.quit()
            # os._exit(0)
            # os._exit(1)
            break

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    pygame.quit()
    # os._exit(0)
    # os._exit(1)
