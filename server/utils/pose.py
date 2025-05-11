import base64
import cv2
import numpy as np
import mediapipe as mp
import logging
import pickle
from data_preprocess import mediapipe_pose_estimation, normalize_joint_coordinates, extract_features

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# load pose model & MultiLabelBinarizer
model_filename = './server/utils/action_recognition_model.pkl'
mlb_filename = './server/utils/multilabel_binarizer.pkl'

with open(model_filename, 'rb') as file:
    classifier = pickle.load(file)

with open(mlb_filename, 'rb') as file:
    mlb = pickle.load(file)

logger = logging.getLogger(__name__)

def matchPoseId(frame):
    joint_coords = mediapipe_pose_estimation(frame, pose)
    
    if joint_coords is not None:
        normalized_joints = normalize_joint_coordinates(joint_coords)
        if normalized_joints is not None:
            features, labels = extract_features(normalized_joints, 0)

            if features is not None:
                # reshape features
                features = features.reshape(1, -1)  # 2d array

                # use MultiOutputClassifier to predict
                y_pred_transformed = classifier.predict(features) 

                # transform into the original label
                y_pred = mlb.inverse_transform(y_pred_transformed) 
                if y_pred:
                    return y_pred[0][0]
    return -1
    
def decode_image(image_data):
    image_data = base64.b64decode(image_data)
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image

class SkillMatch():
    def __init__(self, delay_count):
        self.__counts = 0
        self.__curr_id = -1
        self.delay_count = delay_count
        
    def checkMatchId(self, frame):
        id = matchPoseId(frame)
        if self.__curr_id == id:
            self.__counts += 1
        else:
            self.__counts = 1
            self.__curr_id = id
        if self.__counts == self.delay_count:
            self.__counts = 0
            return id
        return -1
        