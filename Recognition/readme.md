<!--
 * @Date: 2024-11-19 23:05:47
 * @LastEditors: Shujie Han
 * @LastEditTime: 2025-03-18 23:26:05
-->
# Recognition Training and Inference

## Recognition Module Training
For training the recognition module, please refer to the following repository:
[InsightFace](https://github.com/deepinsight/insightface).

## Recognition Model Inference
After training the recognition model, update the following line in `RecogInferenceFlow.py`:

- **Line 33**: Modify the path to the trained model.

Ensure the correct model path is set before running inference.
