# YOLOv8 Blur
[![Nuitka Action](https://github.com/Ultr4Dev/yolov8-blur/actions/workflows/action.yml/badge.svg?branch=main&event=release)](https://github.com/Ultr4Dev/yolov8-blur/actions/workflows/action.yml) # TODO
## Introduction
This is a simple implementation of YOLOv8 with blur augmentation for privacy protection.

## Features
- [x] Real-time object detection with YOLOv8.
- [x] Blur augmentation for privacy protection.
- [x] Virtual camera output for use in applications like Zoom, Skype, and Discord.
## Planned Features
- [ ] Fine-tuning the model for better performance.
- [ ] Face identification for selective blurring.
- [ ] Inpainting to fill the black screen with the background.
- [ ] GUI for easy configuration.
- [ ] 3D Avatar overlay for people in the frame. (Depends on Face identification feature.)

## Requirements
- Python 3.10
- CUDA 12.1
- [OBS Studio (With Virtual Camera)](https://obsproject.com/) or [Unity Capture](https://github.com/schellingb/UnityCapture)

## Usage
1. Install the requirements.
2. Run the following command to start the virtual camera:
    ```bash
    python main.py
    ```
3. Open the application you want to use the virtual camera with.
4. Select the virtual camera as the video input.

Your application should now use the virtual camera with YOLOv8 blur augmentation. TV screens will be black, and people will be blurred.

## Installation
```bash
pip install -r requirements.txt
```

> **Note:** Ensure you install the correct PyTorch and torchvision versions for your CUDA version. You can find the correct version [here](https://pytorch.org/get-started/locally/).

## Notes
- Achieves 11-15 FPS on an RTX 3080 10GB depending on the resolution and number of objects detected.
- The model is trained on the COCO dataset.
- The model may not work perfectly in all scenarios.
- Not optimized for speed.

# Citation
- **Ultralytics YOLOv8**: Glenn Jocher, Ayush Chaurasia, Jing Qiu. (2023). Ultralytics YOLOv8 (v8.0.0). Available at: https://github.com/ultralytics/ultralytics. License: AGPL-3.0.
- **OpenCV**: Intel Corporation, Willow Garage, Itseez. (2023). OpenCV (v4.9.0.80). Available at: https://opencv.org. License: 3-clause BSD.
- **PyTorch**: Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan. (2024). PyTorch (v2.2.2). Available at: https://pytorch.org. License: BSD-3-Clause.
- **TensorFlow**: Martín Abadi, et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Available at: https://www.tensorflow.org.
