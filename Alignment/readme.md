<!--
 * @Date: 2024-11-19 23:07:57
 * @LastEditors: Shujie Han
 * @LastEditTime: 2025-03-18 23:20:55
-->
# Alignment Training and Inference

## Alignment Module Training
For training the alignment module, please refer to the MaskRCNN implementation in the following repository:
[MMCV](https://github.com/open-mmlab/mmcv).

## Alignment Model Inference
After training the alignment model, update the following line in `AligInferenceFlow.py`:

- **Line 74**: Modify the path to the trained model.

Ensure the correct model path is set before running inference.

