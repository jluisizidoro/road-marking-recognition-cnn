# Horizontal Road Marking Recognition for Autonomous Mobility

This repository contains the dataset processing pipeline, deep learning source code, and benchmarking scripts for the paper:

**"Comparison of Neural Networks for Horizontal Road Marking Recognition Aimed at Autonomous Mobility"**


## Overview

Classification of horizontal road markings under real-world driving conditions using transfer learning across VGG16, MobileNetV2, ResNet50, and EfficientNetB0 architectures. The models were evaluated on a curated dataset of 957 real-world frames (320x72 pixels) with an independent test set of 342 images (114 per class under CONTRAN Res. 973/2022 standards).

---

## Repository Structure

```text
road-marking-recognition-cnn/
├── README.md
├── requirements.txt
├── dataset/
│   ├── Treino_615/                  # 615 training frames (205 per class)
│   │   ├── LFO2/
│   │   ├── LFO3/
│   │   └── LMS2/
│   └── Validacao_342/                   # 342 independent test frames (114 per class)
│       ├── LFO2/
│       ├── LFO3/
│       └── LMS2/
└── src/
    ├── framegenerator.py    # Video frame extraction
    ├── frameselector.py     # Temporal frame sampling (1:10 ratio)
    ├── framecutter.py          # ROI cropping (removing upper 3/5)
    ├── neural network training.py # 36-run Grid Search training (5% Dropout)
    ├── multiseedtenattempts.py   # 10-run multi-seed statistical evaluation
    ├── confusionmatrix.py  # Confusion matrix generation
    └── graphgenerators_recallprecisionf1score.py # Precision, Recall, and F1-score chart generator
