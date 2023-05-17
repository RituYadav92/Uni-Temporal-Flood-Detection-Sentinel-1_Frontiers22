[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.4-blue.svg)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/RituYadav92/NuScenes_radar_RGBFused-Detection/blob/master/LICENCE)

# Deep Attentive Fusion network for Flood Detection on Uni-temporal Sentinel-1 Data (Frontiers in Remote Sensing)
[DOI: https://doi.org/10.3389/frsen.2022.1060144]

[GEE App : https://erritu92.users.earthengine.app/view/flooddetectionunitemporal]
## Description: 
Floods are occurring across the globe, and due to climate change, flood events are expected to increase in the coming years. Current situations urge more focus on efficient monitoring of floods and detecting impacted areas. In this study, we propose two segmentation networks for flood detection on uni-temporal Sentinel-1 Synthetic Aperture Radar data. The first network is “Attentive U-Net”. It takes VV, VH, and the ratio VV/VH as input. The network uses spatial and channel-wise attention to enhance feature maps which help in learning better segmentation. “Attentive U-Net” yields 67% Intersection Over Union (IoU) on the Sen1Floods11 dataset, which is 3% better than the benchmark IoU. The second proposed network is a dual-stream “Fusion network”, where we fuse global low-resolution elevation data and permanent water masks with Sentinel-1 (VV, VH) data. Compared to the previous benchmark on the Sen1Floods11 dataset, our fusion network gave a 4.5% better IoU score. Quantitatively, the performance improvement of both proposed methods is considerable. The quantitative comparison with the benchmark method demonstrates the potential of our proposed flood detection networks. The results are further validated by qualitative analysis, in which we demonstrate that the addition of a low-resolution elevation and a permanent water mask enhances the flood detection results. Through ablation experiments and analysis we also demonstrate the effectiveness of various design choices in proposed networks.

<img src="https://github.com/RituYadav92/Uni-Temporal-Flood-Detection-Sentinel-1/blob/main/frsen-03-1060144-g001.jpg" alt="Flood Sites" width="450" height="300">&nbsp; &nbsp; &nbsp;<img src="https://github.com/RituYadav92/Uni-Temporal-Flood-Detection-Sentinel-1/blob/main/GEE_Sample_vis.JPG" alt="GEE App sample visualization" width="450" height="300">

<img src="https://github.com/RituYadav92/Uni-Temporal-Flood-Detection-Sentinel-1/blob/main/Quant_results.JPG" alt="Flood Sites" width="900" height="400">

## Training and Evaluation
Training and evaluation code is in `flood.py`. Before training or evaluating add dataset path, weight path and weight file name.
```
# Train the model
python3 flood.py train 

# Evaluate the model
python3 flood.py evaluate 
```

## Contact Information: 
Ritu Yadav (Email: er.ritu92@gmail.com)

## Cite
Please cite our code if you use it.
      @article{yadav2022deep,
        title={Deep attentive fusion network for flood detection on uni-temporal Sentinel-1 data},
        author={Yadav, Ritu and Nascetti, Andrea and Ban, Yifang},
        journal={Frontiers in Remote Sensing},
        pages={106},
        year={2022},
        publisher={Frontiers}
      }
