<!--
 * @Date: 2025-03-18 23:00:55
 * @LastEditors: Shujie Han
 * @LastEditTime: 2025-03-18 23:17:35
-->
# Detector Training and Inference

## Detector Training
For training the detector, please refer to the following repository:
[YOLOv7 Pose](https://github.com/WongKinYiu/yolov7/tree/pose).

## Model Inference
After training the model, update the following lines in `CompleteInferenceFlow.py`:

- **Line 111**: Modify the path to the trained detector.
- **Line 112**: Modify the path to the test dataset.

Ensure these paths are correctly set before running inference.
