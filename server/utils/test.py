import math
import cv2
import numpy as np
import mediapipe as mp
import time
import pickle

from pose import Pose, isMatchPose, matchPoseId
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


start_time = time.time()
countdown_seconds = 3
countdown_done = False
target = None
user = None

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # Flip the image horizontally for a later selfie-view display, and
        # convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable
        # to pass by reference.
        image.flags.writeable = False
        results = pose.process(image)
                    
        # Draw the pose annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(
        # image,
        # results.pose_landmarks,
        # mp_pose.POSE_CONNECTIONS)
        
        if not countdown_done:

            elapsed = time.time() - start_time
            countdown = countdown_seconds - int(elapsed)

            if countdown > 0:
                cv2.putText(image, str(countdown), (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            elif countdown <= 0 and elapsed < countdown_seconds + 1:
                cv2.putText(image, "GO!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5, cv2.LINE_AA)
            else:
                countdown_done = True
                if results.pose_landmarks:
                    target = Pose(results.pose_landmarks.landmark, 1000)
        else:
            if results.pose_landmarks:
                # result = isMatchPose(image)
                # if result == True:
                #     print("Success")
                # else:
                #     print("Failed")
                result = matchPoseId(image)
                if result != None:
                    print("Match " + result)
                else:
                    print("Match Failed")
                    
        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
