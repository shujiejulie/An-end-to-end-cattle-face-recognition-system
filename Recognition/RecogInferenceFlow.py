# -*- coding: utf-8 -*-
'''
Time: 2022-08-03 19:07:06
Author: Shujie
Content: 
version: 0.0
'''
import torch
from torchvision import transforms
import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('Recognition')
from losses import CombinedMarginLoss
from partial_fc_v2 import PartialFC_V2
from backbones import get_model


@torch.no_grad()
def RecogModel(device):
    # load models.
    backbone = get_model('r100', fp16=False).to(device)
    margin_loss = CombinedMarginLoss(
        10,
        1.0,
        0.0,
        0.4,
        0
    )
    module_partial_fc = PartialFC_V2(margin_loss, 512, 17, 1, False)

    checkpoint = torch.load('/your/recognition/model.pt')
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.eval()

    module_partial_fc.load_state_dict(checkpoint['partial_fc'])
    module_partial_fc.eval().to(device)

    return backbone, module_partial_fc

@torch.no_grad()
def RecogInferenceOneShot(backbone, module_partial_fc, crops, threshold, device):
    # output
    results = {}

    for key, crop in crops.items():
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            ])
        crop = transform(crop)
        
        class17_to_idx = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16}
        idx_to_class17 = {v : k for k, v in class17_to_idx.items()}
        imgs = crop.unsqueeze(0)

        local_embeddings = backbone(imgs.to(device))
        _, logits = module_partial_fc(local_embeddings, torch.tensor(100).to(device))
        
        logits = torch.nn.functional.softmax(logits, dim=1)
        prob, prediction = torch.max(logits, dim=1)

        if prob >= threshold:
            results[key] = [idx_to_class17[int(prediction[0])], prob[0].cpu().detach().numpy()]
    
    return results

@torch.no_grad()
def VerifBetFramesByIOU(pre_bboxes, cur_bboxes, threshold):

    def calculate_iou(box1, box2, threshold):
        """
        Calculate IoU (Intersection over Union) of two bounding boxes.
        
        Args:
            box1: List or tuple containing (xmin, ymin, xmax, ymax) of the first bounding box.
            box2: List or tuple containing (xmin, ymin, xmax, ymax) of the second bounding box.
            
        Returns:
            float: IoU value.
        """
        # Calculate the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of each bounding box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area
        if iou < threshold:
            return 0.0
        else:
            return iou

    cost_iou = np.zeros((len(pre_bboxes), len(cur_bboxes)))
    for i in range(len(pre_bboxes)):
        for j in range(len(cur_bboxes)):
            cost_iou[i][j] = calculate_iou(pre_bboxes[i], cur_bboxes[j], threshold)
    
    row_ind, col_ind = linear_sum_assignment(-cost_iou)
    
    return row_ind, col_ind

@torch.no_grad()
def RecogByPoseCls(backbone, module_partial_fc, cropss, threshold, device):

    min_dist = 0.0
    recog_id = -1
    if len(cropss) == 0:
        return recog_id, min_dist
    
    class17_to_idx = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16}
    idx_to_class17 = {v : k for k, v in class17_to_idx.items()}

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    for i in range(len(cropss)):
        cropss[i] = transform(cropss[i])
    cropss = torch.stack(cropss)
    cropss = cropss.split(200)

    prob_list = []
    pred_list = []
    for crops in cropss:

        local_embeddings = backbone(crops.to(device))
        _, logits = module_partial_fc(local_embeddings, torch.tensor(100).to(device))
        
        logits = torch.nn.functional.softmax(logits, dim=1)
        probs, predictions = torch.max(logits, dim=1)

        probs = list(probs.cpu().detach().numpy())
        predictions = list(predictions.cpu().detach().numpy())
        for i in range(len(probs)):
            if probs[i] >= threshold:
                prob_list.append(probs[i])
                pred_list.append(predictions[i])
 
    if len(pred_list) == 0:
        return recog_id, min_dist

    max_prob = max(prob_list)
    pred = pred_list[prob_list.index(max_prob)]
    min_dist = max_prob
    recog_id = idx_to_class17[pred]
    
    return recog_id, min_dist
