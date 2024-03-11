# VisionLab

Collection of computer vision tools

## Installation

Install this github repo with

```bash
python3 -m pip install -U git+https://github.com/KolinGuo/VisionLab.git
```

## TensorRT Inference

For TensorRT inference, please install the necessary dependencies:

```bash
pip install tensorrt onnx
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt
```

To perform TensorRT inference, you need to compile detection models like Owl(v2) and segmentation models like EfficientVit-SAM into TensorRT engine. See https://github.com/xuanlinli17/nanoowl and https://github.com/xuanlinli17/efficientvit/blob/master/applications/sam.md