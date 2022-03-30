[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.4-blue.svg)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/RituYadav92/NuScenes_radar_RGBFused-Detection/blob/master/LICENCE)

# Deep Attentive Fusion network for Flood Detection on Uni-temporal Sentinel-1 Data
## Description: 
Floods are occurring across the globe and due to climate change, flood events are expected to increase in upcoming years. Current situations urge more focus on efficient monitoring of floods and detecting impacted areas. In this study we propose two architectures for flood detection on uni-temporal Sentinel-1 SAR data. The first network presented in this work is 'Attentive U-Net' where feature maps are enhanced using spatial and channel-wise attention. This network works on VV, VH and the ratio of VV/VH. 'Attentive U-Net' yields 67\% IOU on the Sen1Flood11 dataset, which is 3\% better than the benchmark IOU. The second proposed network is a dual-stream fusion network, where low-resolution elevation data, permanent water mask, and Sentinel-1 SAR (VV, VH) data are fused to extract flood areas. The 'Fusion network' improved IOU by 5\% in comparison to the benchmark IOU. Our code will be open-sourced for reuse and experiment.

<img src="https://github.com/RituYadav92/UNI_TEMP_FLOOD_DETECTION/Result_samples.png" alt="alt text" width="300" height="200">
