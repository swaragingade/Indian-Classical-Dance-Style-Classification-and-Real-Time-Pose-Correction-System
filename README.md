# 💃 NrityaAI — Indian Classical Dance Style Classification & Real-Time Pose Correction

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-00897B?style=for-the-badge&logo=google)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **NrityaAI is a real-time AI system that classifies Indian classical dance styles (Bharatanatyam, Kathak, Odissi) and provides joint-level corrective pose feedback — all from a standard webcam, with no GPU or depth sensor required.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tools & Libraries](#-tools--libraries)
- [Getting Started](#-getting-started)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Pose Correction Module](#-pose-correction-module)
- [Project Structure](#-project-structure)
- [Authors](#-authors)

---

## 🔍 Overview

Indian classical dance forms like **Bharatanatyam**, **Kathak**, and **Odissi** demand precise postural discipline historically taught through in-person instruction. NrityaAI bridges this gap using AI — delivering real-time style classification and personalized corrective feedback through a standard 2D webcam.

The system uses **MediaPipe BlazePose** to extract 33 skeletal landmarks per frame and a **dual-output spatio-temporal CNN-LSTM architecture** to simultaneously predict the dance style and generate joint-level corrective instructions compared against a curated 24-pose reference database.

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| ✅ Test Accuracy | **99.65%** |
| ✅ Macro F1-Score | **0.9965** |
| ✅ Training Samples | **40,977** augmented windows |
| ✅ Dance Styles | **3** (Bharatanatyam, Kathak, Odissi) |
| ✅ Reference Poses | **24** named poses |
| ✅ Hardware Required | Standard CPU + Webcam (no GPU!) |

---

## ✨ Features

- 💃 **Dance Style Classification** — Real-time prediction of Bharatanatyam, Kathak, or Odissi with class-probability distribution
- 🦴 **Pose Keypoint Extraction** — 33 anatomical landmarks extracted per frame using MediaPipe BlazePose
- 📐 **Joint-Level Corrective Feedback** — Angular deviation analysis across 6 key joints with actionable correction messages
- 🏅 **Quality Scoring** — Real-time pose quality score (0–100) compared against expert reference poses
- 🌐 **Web-Based Interface** — Browser-accessible Streamlit frontend for live webcam or pre-recorded video input
- ⚡ **Lightweight Deployment** — Runs entirely on CPU, no depth sensor or GPU required

---

## 🏗️ System Architecture

```
User (Webcam / Video Upload)
        ↓
Presentation Layer — Streamlit Web UI
        ↓
Application Layer — FastAPI REST Gateway
        ↓
Frame Preprocessing → MediaPipe BlazePose
        ↓
    33 × 4 Keypoints per Frame
        ↓
AI Processing Engine
    ├── CNN-LSTM Classification Model → Dance Style
    └── Joint-Angle Quality Scorer   → Feedback Score
        ↓
JSON Response → Display Results (UI)
```

---

## 🛠️ Tools & Libraries

| Tool / Library | Purpose | Link |
|----------------|---------|------|
| 🐍 **Python** | Core scripting and data pipeline | [Download](https://www.python.org/downloads/) |
| 🤸 **MediaPipe BlazePose** | Real-time 33-landmark skeletal extraction | [Docs](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| 🧠 **TensorFlow / Keras** | CNN-LSTM model training and inference | [Docs](https://www.tensorflow.org/) |
| 📷 **OpenCV** | Video capture and frame preprocessing | [Docs](https://opencv.org/) |
| ⚡ **FastAPI** | REST API backend for model serving | [Docs](https://fastapi.tiangolo.com/) |
| 🌐 **Streamlit** | Interactive web frontend | [Docs](https://streamlit.io/) |
| 🔢 **NumPy** | Feature engineering and augmentation | [Docs](https://numpy.org/) |
| 🐼 **pandas** | Dataset management and analysis | [Docs](https://pandas.pydata.org/) |
| 📊 **matplotlib** | Training visualization and reports | [Docs](https://matplotlib.org/) |
| ⚖️ **scikit-learn** | Evaluation metrics and preprocessing | [Docs](https://scikit-learn.org/) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam (for live inference)
- No GPU required!

### Installation

```bash
# Clone the repository
git clone https://github.com/swaragingade/Indian-Classical-Dance-Style-Classification-and-Real-Time-Pose-Correction-System.git
cd Indian-Classical-Dance-Style-Classification-and-Real-Time-Pose-Correction-System

# Install required dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Start the FastAPI backend
uvicorn api.main:app --reload

# In a new terminal, start the Streamlit frontend
streamlit run app/streamlit_app.py
```

Then open your browser at `http://localhost:8501` 🎉

---

## 📋 Project Workflow

### Step 1 — Video Acquisition & Keypoint Extraction
Dance videos are sub-sampled at **10 FPS** and each frame is passed through **MediaPipe BlazePose** to extract 33 skeletal landmarks `(x, y, z, visibility)` per frame.

### Step 2 — Feature Engineering
- **Coordinate Normalisation** — Min-max scaling to `[−1, +1]`
- **Joint Angle Computation** — 6 anatomical angles (elbows, knees, hips) per frame
- **Sliding Window Segmentation** — 60-frame windows with 50% overlap (6 seconds of motion)

### Step 3 — Class Balancing & Augmentation
- Random oversampling to balance all 3 classes
- **Horizontal flip** — mirrors x-coordinates to simulate opposite-facing performers
- **Gaussian noise injection** — σ=0.01 to simulate landmark jitter
- Result: **40,977 augmented training samples**

### Step 4 — Model Training
Dual-output **CNN-LSTM** trained end-to-end with weighted multi-task loss:
```
Loss = 1.0 × CrossEntropy(style) + 0.5 × MSE(quality)
```

### Step 5 — Deployment & Inference
REST API serves predictions; Streamlit frontend delivers real-time style classification and pose corrections overlaid with quality score.

---

## 📊 Dataset

| Style | Raw Windows | After Oversampling |
|-------|------------|-------------------|
| Bharatanatyam | 4,553 | 4,553 |
| Kathak | 3,775 | 4,553 |
| Odissi | 3,024 | 4,553 |
| **Total** | **11,352** | **13,659** → **40,977** (after augmentation) |

- Videos sourced from publicly available YouTube recordings
- Diverse performers, camera angles, lighting conditions, and costumes

---

## 🧠 Model Architecture

```
Input: (60 frames × 33 landmarks × 4 channels)
        ↓
TimeDistributed Conv1D (64 filters, kernel=3)
TimeDistributed Conv1D (128 filters, kernel=3)
TimeDistributed GlobalAveragePooling → (60, 128)
        ↓
LSTM (256 units, return_sequences=True)
Dropout (0.3)
LSTM (128 units, return_sequences=False)
Dropout (0.3)
        ↓
        ├── Style Head: Dense(3, softmax) → Dance Style
        └── Quality Head: Dense(1, sigmoid) → Pose Score
```

---

## 🎯 Pose Correction Module

The system maintains a **24-pose reference database** across 3 styles:

| Style | Poses | Examples |
|-------|-------|---------|
| Bharatanatyam | 11 | Ardhamandalam, Nataraj, Muzhumandi, Samapadam |
| Kathak | 7 | Aamad, Tatkar, Chakkar, Thaat, Namaskar |
| Odissi | 6 | Tribhanga, Chowk, Abhanga, Samabhanga |

At inference, joint angles are compared against the reference. Any joint exceeding **15° deviation** is flagged with directional feedback ("bend more" / "extend more"). A quality score **Q ∈ [0, 100]** is computed proportional to angular deviations.

---

## 📁 Project Structure

```
nrityaai/
│
├── api/                      # FastAPI backend
│   └── main.py               # REST endpoints
├── app/                      # Streamlit frontend
│   └── streamlit_app.py      # Web UI
├── data/                     # Dataset and keypoints
│   └── keypoints/            # Per-style CSV files
├── feature_engineering/      # Preprocessing scripts
├── methods/                  # CNN-LSTM model code
│   ├── gtan/
│   ├── rgtan/
│   └── stagn/
├── models/                   # Saved model checkpoints
├── reports/                  # Evaluation results
├── scripts/                  # Training scripts
├── src/                      # Core pipeline modules
├── requirements.txt          # Python dependencies
├── run_demo.py               # Quick demo script
└── README.md                 # Project documentation
```

## ⚠️ Disclaimer

This project is developed for **educational and cultural preservation purposes**. Dance videos used for training were sourced from publicly available recordings. All rights to the original performances belong to their respective artists.

---

<p align="center">Made with ❤️ to preserve India's classical dance heritage 💃🕺</p>
