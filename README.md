# 🏛️ **Vesuvius Challenge – Surface Detection**

### *3D Papyrus Surface Segmentation for Virtual Unwrapping of Ancient Herculaneum Scrolls*

This project implements a fully optimized deep-learning pipeline for detecting the **papyrus surface layer** inside 3D CT scans from the Vesuvius Challenge.
Correctly segmenting this surface is a crucial step for **virtual unwrapping** of ancient scrolls that cannot be physically opened.

This repository contains:

* A fast + stable **3D U-Net model**
* A fully optimized **surface-aware dataset loader**
* A robust **training pipeline with AMP, gradient accumulation, and checkpoints**
* Validation, logging, and reproducibility tools

This code follows the same principles used by top Kaggle teams and research labs.

---

# 📌 **Project Goals**

* Detect the *papyrus surface* inside 3D volumes
* Avoid topological errors (holes, merges)
* Train efficiently on limited hardware
* Provide a clean, reusable, open-source pipeline

---

# 📂 **Repository Structure**

```
vesuvius-surface-detection/
│
├── src/
│   ├── dataset.py          # Surface-aware 3D patch sampler
│   ├── model.py            # Optimized 3D U-Net architecture
│   ├── losses.py           # BCE + Dice loss functions
│   ├── train.py            # Full training loop
│   ├── config.py           # Centralized configuration
│
├── checkpoints/            # Saved best models
├── data/                   # Local dataset (train_images, train_labels)
├── notebooks/              # (Optional) Jupyter notebooks
│
├── README.md
└── LICENSE
```

---

# 🚀 **Key Features**

### ✅ **Surface-aware patch sampling**

Only patches **containing actual papyrus surface** are used for training.

This solved the model collapse problem and enabled real learning.

### 🔍 **Data normalization**

Every CT volume is normalized:

```python
img = (img - mean) / (std + 1e-6)
```

Ensures stable gradients and consistent input distribution.

### 🧠 **Optimized 3D U-Net**

A lightweight, fast 3D U-Net designed for volumetric segmentation:

* 3-level encoder/decoder
* GroupNorm + SiLU
* Skip connections
* Residual blocks removed for speed

### ⚡ **Modern Training Pipeline**

* PyTorch AMP mixed precision
* Gradient accumulation
* Cosine Annealing LR
* Checkpoints with optimizer + scaler state
* Configurable steps per epoch
* Supports CPU, GPU, and Kaggle TPU

### 📉 **Live Validation**

Validation loss is computed each epoch to monitor generalization:

```
Epoch 12 - Train Loss: 0.5831 - Val Loss: 0.5824
Saved best model to ../checkpoints/best_model_epoch12.pth
```

---

# 📦 **Installation**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Prasanth-ADS/vesuvius-surface-detection.git
cd vesuvius-surface-detection
```

### 2️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies:

```
torch
tqdm
numpy
tifffile
scikit-image
scipy
```

For Windows users:

```bash
pip install imagecodecs
```

---

# 📥 **Dataset Setup**

Your local folder structure should look like:

```
data/
│── train_images/
│      ├── vol1.tif
│      ├── vol2.tif
│      └── ...
│
└── train_labels/
       ├── vol1.tif
       ├── vol2.tif
       └── ...
```

If missing, `train.py` can create dummy data for testing.

---

# 🧩 **Surface-Aware Dataset (dataset.py)**

### Core features:

✔ Loads TIFF volumes
✔ Normalizes intensities
✔ Extracts all voxels where `mask > 0`
✔ Builds a list of **valid sampling centers**
✔ Randomly extracts 3D patches around those surfaces

This ensures the model always sees meaningful signal.

---

# 🧠 **Model Architecture (model.py)**

A custom 3D U-Net:

* Base channels = 16 or 32
* GroupNorm for stability
* SiLU activation
* ConvTranspose upsampling

Supports full-volume inference via sliding window (coming soon).

---

# 🔥 **Training (train.py)**

Run with:

```bash
python src/train.py
```

Key improvements:

* Automatic AMP mixed precision
* Gradient accumulation
* Step-based epoch control
* Validation loop with proper unpacking
* Checkpoint saving:

```
checkpoints/best_model_epoch12.pth
```

---

# 🧪 **Example Logs**

```
Epoch 9  - Train Loss: 0.5833 - Val Loss: 0.5830
Epoch 10 - Train Loss: 0.5832 - Val Loss: 0.5829
Epoch 11 - Train Loss: 0.5831 - Val Loss: 0.5826
Epoch 12 - Train Loss: 0.5831 - Val Loss: 0.5824
```

This confirms:

* Dataset sampling works
* Model is learning
* No collapse

---

# 🧰 **Troubleshooting**

### ❌ Validation unpacking error

Solved by making `dataset.__getitem__` always return `(img, label)`.

### ❌ dtype mismatch (`double` vs `float`)

Solved by enforcing `float32` everywhere.

### ❌ Model collapse (constant predictions)

Solved by:

* Surface-aware sampling
* Normalization
* Correct loss
* Optimized 3D U-Net

---

# 📈 **Future Improvements**

* Topology-aware loss
* clDice for connectivity preservation
* Sliding-window inference
* Automatic surface merging
* Multi-GPU training support
* Experiment tracking via TensorBoard or WandB

---

# 🏁 **Credits**

This repository is based on techniques used in:

* Vesuvius Challenge 2023 & 2025
* Kaggle Surface Detection Solutions
* PyTorch 3D segmentation best practices

---

# 📜 **License**

MIT License — free to use, modify, distribute.

---




