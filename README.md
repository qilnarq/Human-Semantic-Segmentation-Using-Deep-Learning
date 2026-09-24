# Human-Semantic-Segmentation-Using-Deep-Learning

## Overview

This repository contains the implementation and experiments from my Master's thesis on **human semantic segmentation using deep learning**.

The main goal of the project was to train and compare several modern deep learning architectures for separating a person from the background in images and video.

The project focused on a practical computer vision pipeline:

**image collection → manual annotation → dataset preparation → data augmentation → model fine-tuning → segmentation evaluation → video inference**

The dataset was manually annotated using **CVAT**, and several pretrained segmentation models were fine-tuned on the resulting dataset.

The project compares the following architectures:

* **U-Net with ResNet50 encoder**
* **U-Net++ with ResNet50 encoder**
* **DeepLabV3+ with ResNet50 encoder**

The U-Net experiment is documented in the greatest detail in the accompanying training notebook. The other architectures were trained using the same general dataset preparation and fine-tuning approach.

---

## Project Context

This project was completed as part of my Master's thesis.

The work was primarily **applied rather than methodological**: the objective was not to develop a new segmentation architecture, but to prepare a custom dataset, fine-tune established pretrained models, and evaluate their performance on the target human-segmentation task.

This makes the project particularly relevant to practical applications of:

* Computer Vision
* Semantic Segmentation
* Deep Learning
* Transfer Learning
* Image Processing
* Video Segmentation

---

## Task

The task is **semantic segmentation of a human**.

For each input image, the model predicts a pixel-level mask:

* `0` — background
* `1` — person

Unlike object detection, the task does not produce bounding boxes. The goal is to determine the pixels belonging to the person.

### Example task

```text
Input image
     ↓
Segmentation model
     ↓
Binary segmentation mask
     ↓
Person / Background
```

The predicted mask can subsequently be used to isolate the person from the original image or to perform segmentation directly on video frames.

---

## Dataset

A custom dataset was created for the project.

The images were manually annotated using **CVAT**, with the person represented by a pixel-level segmentation mask.

### Dataset statistics

| Property                     |                          Value |
| ---------------------------- | -----------------------------: |
| Training images              |                            557 |
| Validation images            |                             99 |
| Number of foreground classes |                              1 |
| Foreground class             |                       `person` |
| Background class             |                   `background` |
| Annotation type              | Pixel-level segmentation masks |
| Annotation tool              |                           CVAT |

The relatively small dataset size made data augmentation and transfer learning important components of the training pipeline.

---

## Annotation Pipeline

The dataset preparation process consisted of the following steps:

1. Collect images from the target data.
2. Import images into CVAT.
3. Manually annotate the visible person.
4. Export the annotations.
5. Convert the annotations into segmentation masks.
6. Organize images and masks into the training/validation dataset.
7. Apply preprocessing and augmentation during training.

The resulting dataset contains paired:

```text
image → segmentation mask
```

samples.

---

## Data Augmentation

Data augmentation was used to increase the diversity of the training data.

The U-Net training pipeline included:

* Resize
* Elastic transformation
* Rotation
* Horizontal flip
* Vertical flip
* Random brightness/contrast adjustment
* Gaussian noise
* ImageNet normalization

Additional **upper-body crops** were also used to provide the model with additional examples of the target region.

The use of augmentation is particularly relevant for small segmentation datasets. The original U-Net work also demonstrated the importance of using augmentation to train segmentation networks effectively with relatively few annotated images.

---

# Models

Five segmentation architectures were evaluated.

### Training configuration

| Parameter               |                              Value                              |
| ----------------------- | ------------------: | ------------------: | ------------------: |
| Architecture            |               U-Net |             U-Net++ |          DeepLabV3+ |
| Encoder                 |            ResNet50 |            ResNet50 |            ResNet50 |
| Encoder weights         |            ImageNet |            ImageNet |            ImageNet |
| Input size              |           832 × 832 |           640 × 640 |           832 × 832 |
| Batch size              |                   4 |                   4 |                   4 |
| Learning rate           |                1e-4 |                1e-4 |                1e-4 |
| Maximum epochs          |                 150 |                 150 |                 150 |
| Early stopping patience |                   5 |                   5 |                   5 |
| Random seed             |                  42 |                  42 |                  42 |
| Main metric             |                 IoU |                 IoU |                 IoU |
| Hardware                | 2 × NVIDIA Tesla T4 | 2 × NVIDIA Tesla T4 | 2 × NVIDIA Tesla T4 |

The model checkpoint was saved as:

```text
U_net.pth
```

### U-Net result

The best validation IoU obtained in the documented U-Net experiment was:

**IoU = 0.9876**

U-Net uses an encoder/decoder structure with skip connections between corresponding stages, allowing the network to combine high-level semantic information with spatial information needed for precise segmentation.

---

## 2. U-Net++

U-Net++ was also fine-tuned on the same general segmentation task.
The best validation IoU obtained in the documented U-Net++ experiment was:

**IoU = 0.9912**

The model uses the U-Net family of encoder-decoder architectures with redesigned skip connections intended to improve the feature representation between encoder and decoder stages.

The experiment was used to evaluate how a more complex U-Net-family architecture behaves on the custom human-segmentation dataset.

---

## 3. DeepLabV3+

DeepLabV3+ was evaluated as another semantic segmentation architecture.

The implementation used a **ResNet50 encoder**.
The best validation IoU obtained in the documented U-Net++ experiment was:

**IoU = 0.9886**

The model was fine-tuned for binary human segmentation and evaluated using the segmentation mask produced for the person class.


# Experimental Setup

The experiments were performed using GPU acceleration.

The segmentation models were fine-tuned using the custom annotated dataset rather than trained from scratch.

The general experimental pipeline was:

```text
Custom images
      ↓
Manual annotation in CVAT
      ↓
Segmentation masks
      ↓
Train / validation split
      ↓
Preprocessing + augmentation
      ↓
Pretrained segmentation model
      ↓
Fine-tuning
      ↓
Validation
      ↓
Segmentation metrics
      ↓
Image / video inference
```

---

# Evaluation Metrics

Different metrics were used according to the model family.

## Intersection over Union (IoU)

IoU was used for the semantic segmentation models:

* U-Net
* U-Net++
* DeepLabV3+

It is defined as:

```text
IoU = Intersection / Union
```

or:

```text
IoU = |Prediction ∩ Ground Truth|
      ---------------------------
      |Prediction ∪ Ground Truth|
```

An IoU of `1.0` indicates a perfect overlap between the predicted mask and the ground-truth mask.


# Results

## U-Net

The detailed U-Net experiment produced the following result:

| Model       | Encoder  | Input Size | Best Validation IoU |
| -----       | -------- | ---------: | ------------------: |
| U-Net       | ResNet50 |  832 × 832 |          **0.9876** |
| U-Net++     | ResNet50 |  640 × 640 |          **0.9912** |
| DeepLabV3+  | ResNet50 |  832 × 832 |          **0.9886** |

The result represents the best validation IoU observed during the documented training experiment.

---

## Computational Comparison

The following computational measurements were obtained during the model comparison experiments.

| Model       | Parameters | Training Time |  FPS |  Latency | Model Size |
| ----------- | ---------: | ------------: | ---: | -------: | ---------: |
| U-Net       |      ~28 M |      633.96 s |  2.0 | 511.7 ms |     128 MB |
| U-Net++     |      ~28 M |      765.45 s |  1.6 | 617.8 ms |     192 MB |
| DeepLabV3+  |      ~44 M |      636.85 s |  1.9 | 514.0 ms |     105 MB |

---

# Qualitative Analysis

The experiments showed several practical challenges associated with human segmentation.

### 1. Boundary quality

The segmentation masks could contain:

* jagged boundaries;
* small segmentation artifacts;
* inaccuracies around thin body parts;
* imperfect contours around the person.

This is particularly noticeable around complex boundaries such as hair, arms, legs, and clothing.

### 2. Viewpoint generalization

One of the important limitations of the dataset was the limited diversity of viewpoints.

For example, when training images predominantly represented a person from the back, performance could decrease when the model encountered a frontal view.

This demonstrates that a high validation score on a small and relatively homogeneous dataset does not necessarily guarantee strong generalization to substantially different visual conditions.

### 3. Dataset size

The dataset contained a relatively small number of annotated images.

Consequently, model performance can be sensitive to:

* viewpoint;
* pose;
* clothing;
* illumination;
* background;
* scale;
* image composition.

Increasing the diversity of the dataset would therefore be an important direction for further development.

---

# Video Segmentation

The trained models were also applied to video frames.

The inference pipeline can be represented as:

```text
Input video
     ↓
Extract frames
     ↓
Run segmentation model
     ↓
Generate person mask
     ↓
Apply mask to frame
     ↓
Output segmented video
```

This allows the trained segmentation models to be used not only on individual images but also in a video-processing pipeline.

The FPS and latency measurements were used to estimate the practical inference performance of the different architectures.

---

# Technologies

The project used the following technologies:

* Python
* PyTorch
* Segmentation Models PyTorch
* OpenCV
* NumPy
* Albumentations
* Matplotlib
* Jupyter Notebook
* CVAT

---

# Reproducibility

All models experiment use a fixed random seed:

```python
SEED = 42
```

To reproduce the experiment:

1. Prepare the annotated dataset.
2. Place the images and masks into the expected directories.
3. Install the required Python packages.
4. Open the U-Net training notebook.
5. Configure the GPU environment.
6. Run the training pipeline.
7. Evaluate the model on the validation set.
8. Save the trained checkpoint.

---

# Limitations

The main limitations of the project are related to the dataset and experimental scope.

### Limited dataset size

Only a relatively small number of manually annotated images were available.

### Limited visual diversity

The dataset does not cover all possible:

* human poses;
* camera viewpoints;
* clothing types;
* illumination conditions;
* backgrounds.

### Generalization

A model can achieve a high validation IoU while still experiencing difficulties on images that differ substantially from the training distribution.

### No new segmentation architecture

The project does not propose a new neural network architecture or a new segmentation algorithm.

Instead, the contribution is primarily practical:

* construction of a custom annotated dataset;
* preparation of the segmentation pipeline;
* fine-tuning of established architectures;
* evaluation and comparison of several models;
* analysis of computational efficiency;
* application to video segmentation.

---

# Possible Future Improvements

Several directions could improve the system:

* Increase the size of the manually annotated dataset.
* Add more camera viewpoints.
* Include frontal, side and rear views.
* Add more human poses.
* Increase variation in clothing and appearance.
* Include more diverse backgrounds.
* Improve boundary annotation quality.
* Add more difficult examples to the validation set.
* Investigate higher-resolution segmentation.
* Compare additional lightweight segmentation architectures.
* Apply stronger test-time augmentation.
* Improve video inference performance.
* Evaluate models on a completely independent test set.

---

# Academic Context

This repository represents the computer vision component of my Master's thesis.

The work demonstrates a complete applied deep learning workflow for semantic segmentation:

```text
Problem definition
       ↓
Dataset collection
       ↓
Manual annotation
       ↓
Dataset preparation
       ↓
Data augmentation
       ↓
Transfer learning
       ↓
Model fine-tuning
       ↓
Evaluation
       ↓
Model comparison
       ↓
Video inference
```

The project provided practical experience with the complete lifecycle of a computer vision segmentation system, from dataset annotation to deployment-oriented inference.

---

# Key Takeaways

The project demonstrates experience with:

* **Semantic segmentation**
* **Custom dataset creation**
* **CVAT annotation**
* **Pixel-level mask preparation**
* **Transfer learning**
* **Fine-tuning pretrained CNNs**
* **U-Net and U-Net-family architectures**
* **DeepLabV3+**
* **Data augmentation**
* **IoU and segmentation mAP**
* **GPU-based training**
* **Video inference**
* **Computational efficiency analysis**

---

# References

* Ronneberger, O., Fischer, P., Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI, 2015.
