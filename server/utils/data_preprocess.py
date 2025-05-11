import numpy as np

curr_neck = None
prev_neck = None
prev_joint_coords = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_visible_and_in_frame(landmark, visibility_threshold=0.01):
    return (
        landmark.visibility > visibility_threshold and
        0.0 <= landmark.x <= 1.0 and
        0.0 <= landmark.y <= 1.0
    )

def get_neck(left_shoulder, right_shoulder):
    return (left_shoulder + right_shoulder) / 2

def extract_features(joint_coords, label):
    if joint_coords is None or len(joint_coords) < 17:
        print("keypoint is invalid")
        return None

    try:
        angle1 = calculate_angle(joint_coords[11], joint_coords[13], joint_coords[15])
        angle2 = calculate_angle(joint_coords[12], joint_coords[14], joint_coords[16])
        angle3 = calculate_angle(joint_coords[13], joint_coords[11], joint_coords[23])
        angle4 = calculate_angle(joint_coords[14], joint_coords[12], joint_coords[24])
        angle5 = calculate_angle(joint_coords[11], joint_coords[23], joint_coords[25])
        angle6 = calculate_angle(joint_coords[12], joint_coords[24], joint_coords[26])
        angle7 = calculate_angle(joint_coords[23], joint_coords[25], joint_coords[27])
        angle8 = calculate_angle(joint_coords[24], joint_coords[26], joint_coords[28])
        
        features = np.array([angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8])
        labels = []
        for i in range(8):
            labels.append(label)
        return features, labels
    except IndexError:
        print("index error")
        return None 

def mediapipe_pose_estimation(image, pose):
    global prev_neck
    global prev_joint_coords
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if not (is_visible_and_in_frame(landmarks[11]) and is_visible_and_in_frame(landmarks[12]) and is_visible_and_in_frame(landmarks[23]) and is_visible_and_in_frame(landmarks[24])):
            #print("lose some important landmarks")
            return None
        joint_coords = []
        for index in range(len(landmarks)):
            lm = landmarks[index]
            curr_neck = get_neck(np.array([landmarks[11].x, landmarks[11].y]), np.array([landmarks[12].x, landmarks[12].y]))
            if prev_neck is None:
                prev_neck = curr_neck
            if not is_visible_and_in_frame(lm) and 11 <= index <= 25 and prev_joint_coords is not None:
                lm.x = curr_neck[0] + (prev_joint_coords[index][0] - prev_neck[0])
                lm.y = curr_neck[1] + (prev_joint_coords[index][1] - prev_neck[1])
            joint_coords.append([lm.x, lm.y])
        prev_neck = curr_neck
        prev_joint_coords = joint_coords
        return joint_coords
    else:
        return None

def normalize_joint_coordinates(joint_coords):
    if joint_coords is None or len(joint_coords) < 24: 
        print("no 24 coords")
        return None

    # get shoulder midpoint
    mid_shoulder = get_neck(np.array(joint_coords[11]), np.array(joint_coords[12]))

    # calculate body length
    left_hip = np.array(joint_coords[23])
    right_hip = np.array(joint_coords[24])
    mid_hip = (left_hip + right_hip) / 2
    trunk_length = np.linalg.norm(mid_hip - mid_shoulder)

    if trunk_length == 0:
        return None

    # normalize
    normalized_joints = []
    for joint in joint_coords:
        normalized_joint = [(joint[0] - mid_shoulder[0]) / trunk_length,
                            (joint[1] - mid_shoulder[1]) / trunk_length]
        normalized_joints.append(normalized_joint)

    return normalized_joints
