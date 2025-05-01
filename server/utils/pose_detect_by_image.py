import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def getTargetLandmarks(image=None, image_path=None):
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        print("image load failed")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    if results.pose_landmarks:
        # print(results.pose_landmarks.landmark)
        return results.pose_landmarks.landmark
    return None


def getTargetLandmarkList(image_list):
    landmark_list = []
    for image in image_list:
        landmark_list.append(getTargetLandmarks(image, None))
    return landmark_list
