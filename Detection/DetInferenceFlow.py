'''
Date: 2023-08-21 18:03:04
LastEditors: Shujie Han
LastEditTime: 2024-05-01 16:36:16
FilePath: \FaceRecog_video\Detection\DetInferenceFlow.py
'''
# -*- coding: utf-8 -*-
'''
Time: 2023-08-21 18:03:04
Author: Shujie
Content: 
version: 0.0
'''
import torch

import sys
sys.path.append('Detection')
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression, scale_coords, colorstr


def DetModel(weights, data, device):
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(640, s=gs)  # check img_size
    # half precision only supported on CUDA
    model.half()
    # Configure
    model.eval()
    model.model[-1].flip_test = False
    model.model[-1].flip_index = [0, 2, 1, 4, 3]
    # Dataloader
    dataloader = create_dataloader(data,
                                imgsz,
                                1,
                                gs,
                                # batch_size=1,
                                single_cls=False,
                                pad=0.5,
                                rect=True,
                                prefix=colorstr('val: '),
                                kpt_label=True)[0]
    
    return model, dataloader


def DetInferenceOneShot(model, img, shapes, device):
    img = img.to(device).half()
    img /= 255.0
    with torch.no_grad():
        # inference and training outputs
        out, _ = model(img, augment=False)

        lb = []  
        out = non_max_suppression(out, conf_thres=0.2, iou_thres=0.2, labels=lb, multi_label=True, agnostic=False, kpt_label=True, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])
        
        # make outputs
        outputs = []
        for si, pred in enumerate(out):
            # native-space
            scale_coords(img[si].shape[1:], pred[:,:4], shapes[si][0], shapes[si][1], kpt_label=False)
            scale_coords(img[si].shape[1:], pred[:,6:], shapes[si][0], shapes[si][1], kpt_label=True, step=3)
            box = pred[:, :4]

            for p, b in zip(pred.tolist(), box.tolist()):
                outputs.append(
                  {
                    'bbox': [round(x, 3) for x in b],
                    'score': round(p[4], 5),
                    'keypoints': p[6:], 
                  }
                )

    return outputs
