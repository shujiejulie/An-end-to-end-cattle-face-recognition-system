# -*- coding: utf-8 -*-
'''
Time: 2022-09-23 13:48:22
Author: Shujie
Content: 
version: 0.0
'''
from os.path import join

import cv2
import math
import numpy as np
from PIL import Image

from mmdet.apis import init_detector, inference_detector
import mmcv
from mmcv import Config

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)

    image_points = np.array([
                            (landmarks[0], landmarks[1]),     # Nose tip
                            (landmarks[2], landmarks[3]),     # Nose tip
                            (landmarks[4], landmarks[5]),     # Left eye corner
                            (landmarks[6], landmarks[7]),     # Right eye corner
                            (landmarks[8], landmarks[9]),     # Left ear bottom corner
                            (landmarks[10], landmarks[11]),     # Right ear bottom corner
                        ], dtype="double")
                        
    model_points = np.array([ # m -> 0.1mm; reverse z-axis & y-axis
                            (0.0, 0.0, 0.0),                   # Nose tip
                            (0.0, 0.0, 0.0),                   # Nose tip
                            (450, 968, -968),        # Left eye corner
                            (-450, 968, -968),         # Right eye corner
                            (500, 1061, -1094),    # Left ear bottom corner
                            (-500, 1061, -1094),     # Right ear bottom corner
                        ])
  
    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (int(roll), int(pitch), int(yaw)), (int(landmarks[0]), int(landmarks[1]))

def AligModel(device):
    # Specify the path to model config and checkpoint file
    config_file = 'Alignment/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
    checkpoint_file = '/your/alignment/model.pth'
    cfg = Config.fromfile(config_file)
    # modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint_file, device=device)
    model.cfg = cfg

    return model

def AligInferenceOneShot(model, det_outputs, img_path):
    # output
    crops = []

    # load image
    image = mmcv.imread(img_path)
    # m_image = image
    mask = np.zeros(image.shape[:2], dtype = "int")
    # inference mask
    result = inference_detector(model, image)
    bbox_result, segm_result = result    # segm_result: batch faces poly
    # draw mask
    bboxes = np.vstack(bbox_result)
    inds = np.where(bboxes[:, -1] > 0.3)[0]
    for ind in inds:
        # ==========mask background==============
        mask += segm_result[0][ind].astype(int)
        # =======================================
    mask = mask * 255
    m_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    for det_output in det_outputs:
        bbox = det_output['bbox']       # [76.476, 67.911, 65.249, 74.706] xywh image_height:320 (256;32)simage_width:704 (608;48)
        score = det_output['score']      # 0.90369
        keypoints = det_output['keypoints']  # [135.25650024414062, 128.2379150390625, 0.9970703125, ...]
        
        face_dict = {}
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        crop = m_image[ymin:ymax, xmin:xmax]
        if crop.size > 0:
            crop = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
            face_dict['bbox'] = [xmin, ymin, xmax, ymax]
            face_dict['crop'] = crop
            face_dict['keypoints'] = keypoints
            face_dict['score'] = score

            lms_in_one_dict = {'nose': None, 'Leye': None, 'Reye': None, 'LBear': None, 'RBear': None}
            if keypoints[2] > 0.8:
                lms_in_one_dict['nose'] = keypoints[0:2]
            if keypoints[5] > 0.8:
                lms_in_one_dict['Leye'] = keypoints[3:5]
            if keypoints[8] > 0.8:
                lms_in_one_dict['Reye'] = keypoints[6:8]
            if keypoints[11] > 0.8:
                lms_in_one_dict['LBear'] = keypoints[9:11]
            if keypoints[14] > 0.8:
                lms_in_one_dict['RBear'] = keypoints[12:14]
            # select 5 landmarks face.
            if None not in lms_in_one_dict.values():
                face_dict['rotation'] = 'F'
            else:
                face_dict['rotation'] = 'O'
        
            crops.append(face_dict)

    return crops
