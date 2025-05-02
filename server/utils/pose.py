import base64
import math
import cv2
import numpy as np
import mediapipe as mp
import logging
from .pose_detect_by_image import getTargetLandmarks

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
default_target = None

logger = logging.getLogger(__name__)


class Pose():
    def __init__(self, landmarks, threshold):
        self.__data = landmarks
        self.threshold = threshold

        # angles
        self.A11_13_15 = None  # left_shoulder_elbow_wrist
        self.A12_14_16 = None  # right_shoulder_elbow_wrist

        self.A13_11_23 = None  # left_elbow_shoulder_hip
        self.A14_12_24 = None  # right_elbow_shoulder_hip

        self.A11_23_25 = None  # left_shoulder_hip_knee
        self.A12_24_26 = None  # right_shoulder_hip_knee

        self.A23_25_27 = None  # left_hip_knee_ankle
        self.A24_26_28 = None  # right_hip_knee_ankle

        # self.midhip = None

        self.__point = []
        self.angles = []
        self.valid_angles = 0

        self.getAngle()

    # store each point value from json file into __point[]
    def getPointValue(self):
        if self.__data == None:
            for i in range(33):
                self.__point.append([0, 0, 0])
        else:
            for i in range(33):
                self.__point.append([self.__data[i].x,
                                     self.__data[i].y,
                                     self.__data[i].z,
                                     self.__data[i].visibility])

    def is_visible_and_in_frame(self, landmark, visibility_threshold=0.01):
        return (
            landmark[3] > visibility_threshold and
            0.0 <= landmark[0] <= 1.0 and
            0.0 <= landmark[1] <= 1.0
        )

    # calculate the angle by three points
    def calAngle(self, p0, p1, p2):
        if not self.is_visible_and_in_frame(p0) or not self.is_visible_and_in_frame(p1) or not self.is_visible_and_in_frame(p2):
            return -1
        v1 = [p1[0]-p0[0], p1[1]-p0[1]]
        v2 = [p2[0]-p1[0], p2[1]-p1[1]]
        angle = (math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])) / math.pi*180

        if angle < 0:
            angle += 360
        self.valid_angles += 1
        return angle

    # get all the angles in the body
    def getAngle(self):
        self.getPointValue()

        self.A11_13_15 = self.calAngle(self.__point[11], self.__point[13], self.__point[15])
        self.A12_14_16 = self.calAngle(self.__point[12], self.__point[14], self.__point[16])

        self.A13_11_23 = self.calAngle(self.__point[13], self.__point[11], self.__point[23])
        self.A14_12_24 = self.calAngle(self.__point[14], self.__point[12], self.__point[24])

        self.A11_23_25 = self.calAngle(self.__point[11], self.__point[23], self.__point[25])
        self.A12_24_26 = self.calAngle(self.__point[12], self.__point[24], self.__point[26])

        self.A23_25_27 = self.calAngle(self.__point[23], self.__point[25], self.__point[27])
        self.A24_26_28 = self.calAngle(self.__point[24], self.__point[26], self.__point[28])

        self.angles = [self.A11_13_15, self.A12_14_16, self.A13_11_23,
                       self.A14_12_24, self.A11_23_25, self.A12_24_26,
                       self.A23_25_27, self.A24_26_28]

        # midhip_x = (self.__point[8][0] + self.__point[11][0]) / 2
        # midhip_y = (self.__point[8][1] + self.__point[11][1]) / 2
        # midhip_point = []

        # if self.__point[8][2] == 0 or self.__point[11][2] == 0:
        #     midhip_point = [0, 0, 0]
        # else:
        #     midhip_point = [midhip_x, midhip_y, 1]

        # self.midhip = self.calAngle([0, self.__point[1][1], 1], self.__point[1], midhip_point)

    # get pose difference from target pose
    def getPoseDiff(self, target):
        diffSum = 0
        cnt = 0

        for i in range(int(len(self.angles))):
            if self.angles[i] == -1 or target.angles[i] == -1:
                continue
            diff = self.angles[i] - target.angles[i]

            if diff > 180:
                diff = 360 - diff
            diffSum += diff ** 2
            cnt += 1
        # print(cnt)
        # print(self.valid_angles)
        # print(target.valid_angles)
        if cnt == 0 or self.valid_angles < target.valid_angles:
            return 50000
        avgDiff = diffSum / cnt
        # print(str(avgDiff))
        return avgDiff

    # compare whether the pose difference is less than given threshold of the target
    def isMatch(self, target):
        sumOfDiff = self.getPoseDiff(target)
        logger.info(f"sumOfDiff: {sumOfDiff}, threshold: {target.threshold}")
        if sumOfDiff < target.threshold:
            return True
        return False


target_landmarks = getTargetLandmarks(image_path='./server/tsuyu.jpg')
if target_landmarks:
    default_target = Pose(target_landmarks, 1000)

target_paths = ['./server/tsuyu.jpg', './server/littleLeaf.jpg', './server/bigLeaf.jpg']
default_targets = []
for target_path in target_paths:
    target_landmarks = getTargetLandmarks(image_path=target_path)
    if target_landmarks:
        target = Pose(target_landmarks, 1000)
        default_targets.append(target)


def isMatchPose(frame, target=default_target):
    if target == None:
        print("target is Nonetype")
        return False
    # To improve performance, optionally mark the image as not writeable
    # to pass by reference.
    # frame.flags.writeable = False
    # frame.flags.writeable = False
    results = pose.process(frame)

    # Draw the pose annotations on the frame.
    # frame.flags.writeable = True

    if results.pose_landmarks:
        user = Pose(results.pose_landmarks.landmark, 0)
        if user.isMatch(target):
            return True
    return False


def matchPoseId(frame, targets=default_targets):
    # frame.flags.writeable = False
    results = pose.process(frame)
    # frame.flags.writeable = True

    successMatch = []
    if results.pose_landmarks:
        user = Pose(results.pose_landmarks.landmark, 0)
        for i in range(len(targets)):
            if user.isMatch(targets[i]):
                successMatch.append([i, user.getPoseDiff(targets[i])])

    if successMatch:
        best = min(successMatch, key=lambda x: x[1])
        return best[0]

    return -1


def decode_image(image_data):
    image_data = base64.b64decode(image_data)
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image
