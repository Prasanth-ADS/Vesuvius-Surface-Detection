
---

# 📘 **Contributing to Vesuvius Surface Detection**

Thank you for your interest in contributing!
This project aims to build a robust deep-learning pipeline for detecting papyrus surfaces inside 3D CT scans from the Vesuvius Challenge.
Contributions of all kinds — code, documentation, bug fixes, ideas — are welcome.

---

# 🧭 **How to Contribute**

## 1️⃣ **Fork the repository**

Click the **Fork** button on the GitHub page.

## 2️⃣ **Clone your fork**

```bash
git clone https://github.com/<your-username>/vesuvius-surface-detection.git
cd vesuvius-surface-detection
```

## 3️⃣ **Create a new branch**

Use descriptive branch names:

```bash
git checkout -b feature/improve-sampler
```

or

```bash
git checkout -b fix/normalization-bug
```

## 4️⃣ **Make your changes**

Follow project structure:

```
src/
   dataset.py      # Data loading & surface-aware sampling
   model.py        # UNet architecture
   losses.py       # BCE + Dice loss
   train.py        # Training loop & validation
```

Try to follow existing code style and structure.

## 5️⃣ **Test your changes**

Before making a PR, ensure everything works.

### ✔ Test dataset loading:

```bash
python tests/test_dataset_sampling.py
```

### ✔ Test model forward pass:

```bash
python tests/test_model_forward.py
```

### ✔ Run short training loop:

```bash
python src/train.py --epochs 1 --steps-per-epoch 10
```

If you add new features, include tests when relevant.

## 6️⃣ **Commit your changes**

```bash
git add .
git commit -m "Added topology-aware sampling to dataset loader"
```

## 7️⃣ **Push your branch**

```bash
git push origin feature/improve-sampler
```

## 8️⃣ **Open a Pull Request**

Go to your fork → "Compare & Pull Request"
Describe:

* What you changed
* Why the change is needed
* How it improves the project
* Any limitations or remaining issues

---

# 🤝 **Types of Contributions You Can Make**

### 🔧 Bug Fixes

Examples:

* Fix dtype mismatch (float64 vs float32)
* Handle edge-case volumes
* Mask alignment issues
* Missing or NaN voxel handling

### ⚙️ Improvements

* Faster training
* 3D preprocessing speed-ups
* Better patch sampling
* Sliding window inference

### 🤖 Model Enhancements

* Add clDice (connectivity-preserving loss)
* Add topology-aware loss
* Add ViT-UNet hybrids
* Replace GroupNorm with InstanceNorm for specific tasks

### 📚 Documentation

* Clarify README files
* Add diagrams explaining pipeline
* Create tutorials or notebooks

### 🧪 Testing

* Unit tests for dataset logic
* Stress tests for patch alignment
* Evaluate model with toy synthetic data

We appreciate every contribution — large or small.

---

# 📏 **Coding Style Guidelines**

* Use **PEP8** formatting.
* Use 4 spaces for indentation.
* Add comments for non-trivial logic.
* Keep functions small and modular.
* Prefer descriptive variable names.

---

# 🔬 **Pull Request Guidelines**

To ensure maintainability:

* One feature or fix per PR
* Avoid mixing refactoring with new features
* Include tests whenever applicable
* Keep PRs concise
* Reference related issues:

  ```
  Fixes #12
  Related to #8
  ```

---

# 📣 **Code of Conduct**

Be respectful, collaborative, and constructive.
All contributors are welcome regardless of background or experience.

This project exists to help bring back to life ancient texts lost for 2,000 years — let’s build something meaningful together.

---

# 🏛️ **Thank You**

Your contributions help improve open-source tools for digital archaeology and historical preservation.
Whether fixing a typo or adding a new UNet architecture, you're helping unlock history.

If you have questions, open an issue or start a discussion!

---
