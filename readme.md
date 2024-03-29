# YOLOv8 Blur
## Introduction
Simple implementation of YOLOv8 with blur augmentation for privacy protection.

## Requirements
- Python 3.10
- CUDA 12.1
- [OBS Studio (With Virtual Camera)](https://obsproject.com/) or [Unity Capture](https://github.com/schellingb/UnityCapture)

## Usage
1. Install the requirements.
2. Run the following command to start the virtual camera.
```bash
python main.py
```
3. Open whatever application you want to use the virtual camera with.
4. Select the virtual camera as the video input.
Your application should now be using the virtual camera with the YOLOv8 blur augmentation. Your TV screens should be black and people should be blurred.

## Installation
```bash
pip install -r requirements.txt
```

> #  Make sure to install the correct version of PyTorch and torchvision for your CUDA version. You can find the correct version [here](https://pytorch.org/get-started/locally/).

