import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T

# Load the pre-trained model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# List of keypoints
keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
             'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

# Function to preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    return img, img_tensor

# Function to draw keypoints
def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    cmap = plt.get_cmap('rainbow')
    img_copy = img.copy()
    color_id = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
    for person_id in range(len(all_keypoints)):
        if confs[person_id] > conf_threshold:
            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]
            for kp in range(len(scores)):
                if scores[kp] > keypoint_threshold:
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1]) * 255)
                    cv2.circle(img_copy, keypoint, 5, color, -1)
    return img_copy

# Function to get limbs from keypoints
def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
    ]
    return limbs

# Function to draw skeleton
def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    cmap = plt.get_cmap('rainbow')
    img_copy = img.copy()
    limbs = get_limbs_from_keypoints(keypoints)
    if len(all_keypoints) > 0:
        colors = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
        for person_id in range(len(all_keypoints)):
            if confs[person_id] > conf_threshold:
                keypoints = all_keypoints[person_id, ...]
                for limb_id in range(len(limbs)):
                    limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                    limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                    limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                    if limb_score > keypoint_threshold:
                        color = tuple(np.asarray(cmap(colors[person_id])[:-1]) * 255)
                        cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 2)
    return img_copy

# Function to perform pose estimation
def estimate_pose(img_tensor):
    with torch.no_grad():
        output = model([img_tensor])[0]
    return output
