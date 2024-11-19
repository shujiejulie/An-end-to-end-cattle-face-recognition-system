# -*- coding: utf-8 -*-
'''
Time: 2023-08-21 17:58:38
Author: Shujie
Content: 
version: 0.0
'''
import mmcv
import cv2
import numpy as np
from os.path import join, exists
from os import makedirs

from Detection.DetInferenceFlow import DetModel, DetInferenceOneShot
from Alignment.AligInferenceFlow import AligModel, AligInferenceOneShot
from Recognition.RecogInferenceFlow import VerifBetFramesByIOU, RecogModel, RecogByPoseCls

def hex_to_rgb(hex_code):
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return (b, g, r)

def imshow_results(img, bboxes, labels, keypoints, out_file):

    colors = ["33ddff", "fa3253", "34d1b7", "ff007c", "ff6037", "ddff33", "24b353", "b83df5", "66ff66", "32b7fa", "ffcc33", "83e070", "fafa37", "5986b3", "8c78f0", "ff6a4d", "f078f0"]

    for bbox, label, keypoint in zip(bboxes, labels, keypoints):
        color_ind = int(label) - 1
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, color=hex_to_rgb(colors[color_ind]), thickness=3)

        if len(keypoint) != 0:
            steps = 3
            skeleton = [[1, 2], [1, 3], [2, 4], [3, 5], [2, 3]]
            for sk_id, sk in enumerate(skeleton):
                r, g, b = (0,0,0)
                pos1 = (int(keypoint[(sk[0]-1)*steps]), int(keypoint[(sk[0]-1)*steps+1]))
                pos2 = (int(keypoint[(sk[1]-1)*steps]), int(keypoint[(sk[1]-1)*steps+1]))
                if steps == 3:
                    conf1 = keypoint[(sk[0]-1)*steps+2]
                    conf2 = keypoint[(sk[1]-1)*steps+2]
                    if conf1<0.8 or conf2<0.8:
                        continue
                if pos1[0]%1920 == 0 or pos1[1]%1080==0 or pos1[0]<0 or pos1[1]<0:
                    continue
                if pos2[0] % 1920 == 0 or pos2[1] % 1080 == 0 or pos2[0]<0 or pos2[1]<0:
                    continue
                cv2.line(img, pos1, pos2, (int(r), int(g), int(b)), thickness=3)
            num_kpts = len(keypoint) // steps
            for kid in range(num_kpts):
                x_coord, y_coord = keypoint[steps * kid], keypoint[steps * kid + 1]
                if not (x_coord % 1920 == 0 or y_coord % 1080 == 0):
                    conf = keypoint[steps * kid + 2]
                    if conf < 0.8:
                        continue
                    cv2.circle(img, (int(x_coord), int(y_coord)), 8, hex_to_rgb(colors[color_ind]), -1)
            
        label_text = f'ID:{label}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 6), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=hex_to_rgb(colors[color_ind]), thickness=2)
    
    mmcv.image.imwrite(img, out_file)

def select_probe(probe, crops):
    for crop in crops:
        if crop['TemID'] not in probe.keys():
            probe[crop['TemID']] = {'F':[], 'P':[]}

        if crop['rotation'] == 'F':
            probe[crop['TemID']]['F'].append(crop['crop'])
        else:
            probe[crop['TemID']]['P'].append(crop['crop'])
    
    return probe, crops

def face_verification(pre_faces, cur_faces, TemID, probe_faces):
        cur_bboxes = [cur_face['bbox'] for cur_face in cur_faces]
        pre_bboxes = [pre_face['bbox'] for pre_face in pre_faces]
        pre_ind, cur_ind = VerifBetFramesByIOU(pre_bboxes=pre_bboxes, cur_bboxes=cur_bboxes, threshold=0.5)
        
        for ind, cur_face in enumerate(cur_faces):
            if ind not in cur_ind:
                TemID += 1
                cur_face['TemID'] = TemID
            else:
                cur_face['TemID'] = pre_faces[pre_ind[list(cur_ind).index(ind)]]['TemID']
        
        probe_faces, cur_faces = select_probe(probe=probe_faces, crops=cur_faces)
        
        return cur_faces, TemID, probe_faces

def face_recognition_cls(probe_faces, backbone, module_partial_fc):
    tem_probe_ids = {}
    for tid, c in probe_faces.items():
        recog_id = -1
        recog_dist = 0.0
        
        recog_id, recog_dist = RecogByPoseCls(backbone=backbone, module_partial_fc=module_partial_fc, cropss=c['F'], threshold=0.95, device='cuda:0') 
        if recog_id == -1:
            recog_id, recog_dist = RecogByPoseCls(backbone=backbone, module_partial_fc=module_partial_fc, cropss=c['P'], threshold=0.99, device='cuda:0')

        tem_probe_ids[tid] = (recog_id, recog_dist)

    return tem_probe_ids 


if '__main__' == __name__:

    DET_WEIGHTS = "/your/detection/model.pt"
    DET_DATA = [
        "/your/dataset/images_path/1",
        "/your/dataset/images_path/2",
        ]
    for DATA in DET_DATA:
        det_model, det_dataloader = DetModel(weights=DET_WEIGHTS, data=DATA, device='cuda:0')
        alig_model = AligModel(device='cuda:0')
        recog_backbone, module_partial_fc = RecogModel(device='cuda:0')
        
        # Cattle Face Alignment: First, detect the face, then save the pose information as {'F', 'P'}.
        all_faces = {} # {file_path: [crops]}
        probe_faces = {} # {TemID: {'F':[crop, ...], 'P':[crop, ...]}, ...}
        TemID = 0
        flag = 0

        for img, targets, paths, shapes in det_dataloader:
            det_outputs = DetInferenceOneShot(model=det_model, img=img, targets=targets, paths=paths, shapes=shapes, device='cuda:0')
            crops = AligInferenceOneShot(model=alig_model, det_outputs=det_outputs, img_path=paths[0])

            if len(crops) != 0:
                if flag == 0:
                    for ind, crop in enumerate(crops):
                        TemID += 1
                        crop['TemID'] = TemID
                    probe_faces, crops = select_probe(probe=probe_faces, crops=crops)
                else:
                    # Cattle Face Verification: Determine which cattle faces belong to the same individual over a period of time (video duration).
                    crops, TemID, probe_faces = face_verification(pre_faces=list(all_faces.values())[-1], cur_faces=crops, recog_backbone=recog_backbone, TemID=TemID, probe_faces=probe_faces) 

                all_faces[paths[0]] = crops
                flag += 1
        
        # Cattle Face Recognition: Select the best sample for each individual to determine its ID.
        tem_probe_ids = face_recognition_cls(probe_faces=probe_faces, backbone=recog_backbone, module_partial_fc=module_partial_fc)

        # imshow
        for file_path, crops in all_faces.items():
            # txt path
            output_path = '/your/output/path/{}_{}_{}'.format(file_path.split('/')[-1].split('.')[0].split('_')[0], file_path.split('/')[-1].split('.')[0].split('_')[1], file_path.split('/')[-1].split('.')[0].split('_')[2])
            det_path = join(output_path, 'det')
            img_path = join(output_path, 'img')
            if not exists(det_path):
                makedirs(det_path)
            if not exists(img_path):
                makedirs(img_path)
            # write txt
            with open(join(det_path, file_path.split('/')[-1].split('.')[0]+'.txt'), 'a') as f:
                img = mmcv.imread(file_path)
                bboxes = []
                labels = []
                keypoints = []
                for crop in crops:
                    
                    tl = crop['TemID']
                    l = tem_probe_ids[tl][0]

                    bbs = crop['bbox']
                    bbs.append(tem_probe_ids[tl][1])
                    w = bbs[2] - bbs[0]
                    h = bbs[3] - bbs[1]
                    cx = (bbs[0] + w/2) / 1920
                    cy = (bbs[1] + h/2) / 1080
                    rw = w / 1920
                    rh = h / 1080

                    labels.append(l)
                    bboxes.append(bbs)
                    keypoints.append(crop['keypoints'])

                    f.write('{} {} {} {} {} {}\n'.format(l, crop['score'], cx, cy, rw, rh))
                
            imshow_results(img=img, bboxes=np.array(bboxes), labels=np.array(labels), keypoints=np.array(keypoints), out_file=join(img_path, file_path.split('/')[-1]))
