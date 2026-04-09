# Skin Cancer Detection using CNN

A deep learning project for binary skin cancer classification (Benign vs. Malignant) using an optimized convolutional neural network.

---

## Overview

This repository contains code, model files, and dataset structure for training and deploying a skin cancer detection model using TensorFlow/Keras. The project includes:

- CNN training pipeline
- Model evaluation and diagnostics
- Web application entry point
- Example datasets for training and testing

---

## Features

- Improved model architecture with batch normalization and global average pooling
- Balanced data augmentation for medical images
- Training diagnostics: accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix
- GPU acceleration support for faster training
- Automatic best-model checkpointing

---

## Repository Contents

- `main.py` - Web application / inference script
- `skin-cancer-detection-with-cnn-deep-learning.ipynb` - Notebook for training and analysis
- `skin_cancer_cnn.h5` - Pretrained model file
- `skin_cancer_cnn_best.h5` - Best saved model during training
- `dataset/` - Image dataset split into training, validation, and test sets
- `static/`, `templates/` - Web app assets and templates

---

## Project Structure

```text
.
├── dataset/
│   ├── train/
│   │   ├── Benign/
│   │   └── Malignant/
│   ├── test/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── validation/
├── static/
│   └── styles.css
├── templates/
│   └── index.html
├── main.py
├── skin_cancer_cnn.h5
├── skin_cancer_cnn_best.h5
├── skin_cancer_detection_script.py
├── skin-cancer-detection-with-cnn-deep-learning.ipynb
├── styles.css
├── README.md
└── gpucheck.py
```

---

## Installation

### 1. Create a Python environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install the core packages manually:

```bash
pip install tensorflow opencv-python streamlit matplotlib numpy
```

### 3. Download the HAM10000 dataset

1. Download the HAM10000 dataset from a trusted source.
2. Extract the images into the `dataset/` folder.
3. Split images into `train/`, `validation/`, and `test/` directories with `Benign/` and `Malignant/` subfolders.

---

## Usage

### Train the model

Open the notebook and run the training pipeline:

```bash
jupyter notebook "skin-cancer-detection-with-cnn-deep-learning.ipynb"
```

### Run the web interface

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Dataset Format

The dataset should follow this structure:

```text
dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
├── test/
│   ├── Benign/
│   └── Malignant/
└── validation/
```

---

## Recommended Setup

### GPU (recommended)

For best performance, use a compatible NVIDIA GPU with CUDA and cuDNN installed.

Verify TensorFlow GPU availability:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### CPU

CPU training is supported but significantly slower.

---

## Evaluation

The model evaluation includes:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

---

## Notes

This project addresses common issues in skin cancer model training, such as:

- inappropriate augmentation for medical images
- unstable CNN architecture design
- insufficient training and checkpointing

---

## Future Improvements

- Add transfer learning with modern backbones (ResNet, EfficientNet)
- Improve dataset balancing and preprocessing
- Add explainability with Grad-CAM or saliency maps
- Deploy as a production-ready web service

---

## License

Specify your license here, for example:

`MIT License`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should **not** be used for medical diagnosis.