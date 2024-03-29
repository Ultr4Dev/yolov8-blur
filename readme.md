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


## Notes
- 11-15 FPS on a RTX 3080 10GB.
- The model is trained on the COCO dataset.
- The model is not perfect and may not work in all scenarios.
- The model is not optimized for speed.

# Citation
- **Ultralytics YOLOv8**: Glenn Jocher, Ayush Chaurasia, Jing Qiu. (2023). Ultralytics YOLOv8 (v8.0.0). Available at: https://github.com/ultralytics/ultralytics. License: AGPL-3.0.

- **OpenCV**: Intel Corporation, Willow Garage, Itseez. (2023). OpenCV (v4.9.0.80). Available at: https://opencv.org. License: 3-clause BSD.

- **PyTorch**: Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan. (2024). PyTorch (v2.2.2). Available at: https://pytorch.org. License: BSD-3-Clause.

- **TensorFlow**: Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, Xiaoqiang Zheng. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Available at: https://www.tensorflow.org.