# MNIST Digit Recognition - Full Stack Application

A complete machine learning project that demonstrates how to build, train, and deploy a handwritten digit classifier using **PyTorch**, **FastAPI**, and **Flask**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Flask]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Architecture](#model-architecture)
- [Screenshots](#screenshots)
- [Learning Resources](#learning-resources)

---

## 🎯 Overview

This project is a teaching resource for **Day 6** that covers:

1. **Machine Learning**: Training a neural network on the MNIST dataset
2. **Backend Development**: Creating a REST API with FastAPI
3. **Frontend Development**: Building a HTML UI for predictions
4. **Full Stack Integration**: Connecting all components together

The MNIST dataset contains 70,000 images of handwritten digits (0-9), each 28x28 pixels in grayscale.

---


## ✨ Features

### Backend (FastAPI)

- 🚀 Fast and modern Python API framework
- 📤 Image upload endpoint for predictions
- 🔄 CORS enabled for frontend communication
- 📖 Auto-generated Swagger documentation
- ⚡ GPU support when available

### Frontend (React + Vite)

- 🖼️ Image upload with preview
- 🔮 Real-time digit predictions
- 📊 Confidence score display
- 🎨 Simple and clean UI

### Model (PyTorch)

- 🧠 Fully connected neural network
- 📈 ~97% accuracy on MNIST test set
- 💾 Saved model weights for inference

---

## 📦 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- Basic understanding of Python and JavaScript

---

## 🛠️ Installation

### 1. Clone or Download the Project

```bash
cd "day 6"
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```


---

## 🚀 Usage

### Step 1: Start the Backend API

Open a terminal and run:

```bash
# From the project root directory
python app.py
```

Or use uvicorn directly:

```bash
uvicorn mnist_api:app --reload
```

### Step 3: Make Predictions

1. Open the frontend in your browser
2. Upload an image of a handwritten digit
3. Click "Predict" to see the result
4. View the predicted digit and confidence score

---
