# Multi-Layer Perceptron from Scratch 

A fully hand-coded Multi-Layer Perceptron (MLP) built using **only NumPy** — no PyTorch, no TensorFlow. Trained on the classic Iris flower dataset to classify 3 species with **96.7% test accuracy**.

---

## Architecture

```
Input Layer (4)  →  Hidden Layer 1 (16, ReLU)  →  Hidden Layer 2 (8, ReLU)  →  Output Layer (3, Softmax)
```

| Layer | Neurons | Activation |
|---|---|---|
| Input | 4 (one per feature) | — |
| Hidden 1 | 16 | ReLU |
| Hidden 2 | 8 | ReLU |
| Output | 3 (one per class) | Softmax |

---

## Dataset

**Iris Flowers** — 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica)

- Training set: 120 samples
- Test set: 30 samples
- Features: sepal length, sepal width, petal length, petal width

---

## Results

| Split | Accuracy |
|---|---|
| Train | **98.3%** |
| Test | **96.7%** |

Training converged smoothly over 1000 epochs with cross-entropy loss dropping from ~0.26 to ~0.04.

---

## What's Implemented from Scratch

- **Z-score normalization** — zero mean, unit variance per feature
- **One-hot encoding** — converts integer labels to binary vectors
- **He weight initialization** — optimal for ReLU activations
- **ReLU activation** + its derivative for backprop
- **Softmax activation** — numerically stable with max subtraction
- **Cross-entropy loss** — standard for multi-class classification
- **Full backpropagation** — chain rule applied layer by layer
- **Gradient descent** — weight and bias updates with configurable learning rate
- **Training loop** with epoch-wise loss and accuracy logging
- **Loss curve plot** using Matplotlib

---

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

Open and run `MLP.ipynb` in Jupyter:

```bash
jupyter notebook MLP.ipynb
```

Or run all cells top to bottom — the notebook is self-contained.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden Layer 1 size | 16 |
| Hidden Layer 2 size | 8 |
| Learning rate | 0.05 |
| Epochs | 1000 |
| Weight init | He initialization |
| Random seed | 42 |

---

## Dependencies

- `numpy` — all math and matrix operations
- `matplotlib` — loss curve visualization
- `scikit-learn` — only used to load the Iris dataset and split data

---

## Key Concepts Demonstrated

- **Forward pass**: input flows through layers to produce class probabilities
- **Backward pass**: gradients flow in reverse to update weights
- **Softmax + Cross-Entropy**: combined for stable gradient computation at the output
- **ReLU**: avoids vanishing gradients in hidden layers

---

## File Structure

```
MLP.ipynb    ← Main notebook (data loading, model, training, evaluation, plot)
README.md    ← This file
```
