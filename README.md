# Custom Single Shot MultiBox Detector (SSD) using MobileNetV2

A lightweight, end-to-end custom Object Detection pipeline built using TensorFlow/Keras and FiftyOne. This project trains a tailored Single Shot Detector (SSD) head on top of a transfer-learned MobileNetV2 backbone to detect two specific target classes: **Persons** and **Cars**.

The system features highly optimized parallel data streaming via `tf.data`, custom anchor-based spatial feature mapping, and Non-Maximum Suppression (NMS) post-processing.

---

## 🚀 Key Features

* **Automated Dataset Pipeline:** Streamlined ingestion, sub-sampling, and class-balancing using the Open Images V6 dataset via the FiftyOne Zoo.
* **Hardware-Optimized Streaming:** Uses `tf.data.Dataset` multi-threading asynchronously (`AUTOTUNE`) to perform on-the-fly decoding, resizing, and float normalization.
* **Custom Object Detection Head:** Replaces standard classification layers with parallel 2D-Convolutional Local Spatial Detectors for regression (bounding boxes) and classification (labels).
* **Advanced Evaluation Suite:** Automated generation of Huber/Cross-Entropy convergence graphs, Scikit-Learn classification reports, and automated Confusion Matrix heatmaps.
* **NMS Post-Processing:** Built-in Non-Maximum Suppression (`tf.image.non_max_suppression`) to filter overlapping predictions and eliminate background noise.

---

## 🏗️ Architectural Design & Implementation

The architecture leverages a hybrid transfer-learning approach combined with a customized anchor-free MultiBox prediction layer.

### 1. Feature Extractor (Backbone)
* **Base Network:** MobileNetV2 pre-trained on ImageNet.
* **Fine-Tuning Strategy:** Partial freezing. The foundational layers (edges, textures) are locked, while the top **40 layers** are unfrozen to allow the network to specialize in the geometries of custom target classes.

### 2. Multi-Task Detection Heads
The network splits into two simultaneous functional heads handling spatial outputs up to 10 object slots per image:
* **Bounding Box Regression:** Employs a 2D Convolutional layer mapped via a Global Average Pool upsampling matrix to output 4 normalized coordinate vectors $(y_{min}, x_{min}, y_{max}, x_{max})$ per object slot using a Sigmoid activation.
* **Classification:** Maps class logits directly through a Softmax layer across the targeted classes.

---

## 🛠️ Installation & Setup

### Prerequisites
Before running the pipeline, ensure you have your Kaggle API Token available for dataset authentication. 

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
