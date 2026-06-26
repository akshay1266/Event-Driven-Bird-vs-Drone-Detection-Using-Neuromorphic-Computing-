# Event-Driven-Bird-vs-Drone-Detection-Using-Neuromorphic-Computing-
# Event-Driven Bird vs Drone Detection using Neuromorphic Computing

A lightweight, from-scratch implementation of a **Spiking Convolutional Neural Network (CSNN)** for bird vs drone classification using **neuromorphic computing**. The project leverages **Leaky Integrate-and-Fire (LIF)** neurons, rate-based spike encoding, and surrogate gradient learning to perform energy-efficient aerial image classification suitable for edge devices.

---

## Table of Contents

* Overview
* Highlights
* Model Architecture
* Data Pipeline
* Training & Optimization
* Hyperparameters
* Results
* Model Efficiency
* Future Improvements

---

## Overview

This project explores **neuromorphic computing** as an alternative to conventional deep learning for aerial object classification. Instead of processing continuous-valued activations like traditional CNNs, the model represents information as sparse temporal spike trains and learns using a **Convolutional Spiking Neural Network (CSNN)**.

The primary objective is to distinguish **birds** from **drones** while maintaining a compact model suitable for low-power edge hardware.

---

## Highlights

* Built completely using **Spiking Neural Networks (SNNs)** with **snnTorch**.
* Event-driven computation using **Leaky Integrate-and-Fire (LIF)** neurons.
* Rate-based spike encoding with temporal processing over **20 time steps**.
* Learnable membrane decay (β) and firing threshold (Vth).
* Surrogate gradient descent for end-to-end training.
* Lightweight model (~60 KB), over **146× smaller** than MobileNetV2.
* Achieved **81.3% test accuracy** on the Bird vs Drone dataset.

---

## Model Architecture

The network consists of:

### Input Encoding

* Rate-based spike encoding
* Temporal window: **20 time steps**

### Feature Extraction

* Conv2D → BatchNorm → LIF → MaxPool
* Conv2D → BatchNorm → LIF → MaxPool
* Conv2D → BatchNorm → LIF → MaxPool

### Classification

* Fully Connected + BatchNorm + LIF
* Dropout (0.5)
* Fully Connected Output + LIF

Final predictions are obtained by accumulating output spikes across all time steps.

---

## Data Pipeline

Images are:

* Resized to **128 × 128**
* Normalized using ImageNet statistics
* Augmented with:

  * Random horizontal flip
  * Random vertical flip
  * Random rotation (±15°)
  * Brightness & contrast jitter

Supported spike encoding methods:

* Rate Encoding *(used for all experiments)*
* Latency Encoding
* Direct Encoding

Additional temporal jitter augmentation improves robustness by randomly shifting spike timings.

---

## Training & Optimization

The network was trained using several optimizations:

* AdamW optimizer
* Fast Sigmoid surrogate gradients
* Cosine Annealing Learning Rate Scheduler
* Gradient clipping
* Automatic checkpoint saving
* GPU-based spike encoding
* BatchNorm before every LIF layer for stable membrane dynamics

---

## Hyperparameters

| Parameter          | Value                     |
| ------------------ | ------------------------- |
| Image Size         | 128 × 128                 |
| Time Steps         | 20                        |
| Batch Size         | 16                        |
| Epochs             | 50                        |
| Optimizer          | AdamW                     |
| Learning Rate      | 1e-3                      |
| Weight Decay       | 1e-4                      |
| Scheduler          | Cosine Annealing          |
| Minimum LR         | 1e-5                      |
| Dropout            | 0.5                       |
| Gradient Clip      | 1.0                       |
| Surrogate Gradient | Fast Sigmoid (Slope = 15) |
| Encoding           | Rate Encoding             |

---

## Results

### Performance

* **Test Accuracy:** **81.3%**
* Training epochs: **50**
* Automatic best-model checkpointing

### Model Characteristics

* Temporal evidence accumulation through spike counts
* Learnable membrane dynamics
* Interpretable membrane potential evolution
* Robust spike-based decision making

---

## Model Efficiency

| Model           |  Accuracy |        Size |
| --------------- | --------: | ----------: |
| EfficientNet-B0 |     92.0% |    15.63 MB |
| MobileNetV2     |     87.0% |     8.76 MB |
| **Our CSNN**    | **81.3%** | **0.06 MB** |

Despite a small reduction in accuracy, the proposed CSNN is:

* **146× smaller than MobileNetV2**
* **260× smaller than EfficientNet-B0**

making it highly suitable for embedded and edge AI applications.

---

## Future Improvements

* Increase spike window to 50–100 time steps.
* Explore latency-based spike encoding.
* Train directly on event-camera datasets.
* Benchmark energy consumption on neuromorphic hardware (Intel Loihi, BrainScaleS).
* Investigate recurrent SNNs and attention-based spike aggregation.
