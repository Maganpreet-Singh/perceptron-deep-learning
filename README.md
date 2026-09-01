<div align="center">

# 🧠 Perceptron — Deep Learning From Scratch

### A hands-on implementation of the fundamental building block behind neural networks

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p>
  <strong>Understand the Perceptron → understand the foundations of neural networks.</strong>
</p>

</div>

---

## 📌 Overview

This repository is a practical deep-learning learning project focused on the **Perceptron**, one of the earliest and simplest models for binary classification.

The implementation uses a Jupyter Notebook and a small placement dataset to explore how a linear neuron can combine input features, apply weights and bias, produce a prediction, and evaluate classification performance.

The project covers the core ideas behind an artificial neuron, including:

- Weighted inputs
- Bias
- Linear decision boundaries
- Activation functions
- Perceptron learning and prediction
- Binary classification
- Train/test splitting
- Model evaluation
- Confusion matrices and classification reports
- Decision-region visualization

This is intentionally a **fundamentals-first** project: the goal is not to hide the mathematics behind a high-level deep-learning framework, but to make the mechanics of a basic neural model easy to inspect and understand.

---

## 🎯 Learning Objectives

By working through this repository, you can build intuition for:

1. **How an artificial neuron works** — inputs are multiplied by weights and combined with a bias.
2. **How classification decisions are made** — the model separates data using a learned linear boundary.
3. **How a model learns** — weights are adjusted according to the Perceptron learning rule.
4. **Why bias matters** — it shifts the decision boundary instead of forcing it through the origin.
5. **How supervised learning is evaluated** — predictions can be compared against known labels using standard classification metrics.
6. **How visualization helps** — decision regions make the learned classifier easier to reason about.

---

## 🧮 Perceptron Fundamentals

A Perceptron receives an input vector and computes a weighted sum:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

A threshold-style activation then converts the score into a class prediction:

```text
ŷ = activation(z)
```

For a binary classification problem, the model learns a linear decision rule. In two dimensions, that rule can be visualized as a straight decision boundary separating the two classes.

### 🔁 Learning intuition

At a high level, the training loop is:

```text
Input data
    ↓
Weighted sum + bias
    ↓
Activation / prediction
    ↓
Compare with target
    ↓
Update weights and bias
    ↓
Repeat across training samples
```

This simple mechanism is historically important because it introduces the same basic ingredients that appear throughout neural-network learning: **parameters, activations, predictions, errors, and parameter updates**.

---

## 📊 Dataset

The repository includes `placement.csv`, a compact binary-classification dataset with **100 rows** and three columns:

| Feature / Target | Description |
|---|---|
| `cgpa` | Student CGPA feature |
| `resume_score` | Resume score feature |
| `placed` | Binary placement target (`0` or `1`) |

The notebook loads the dataset with Pandas and performs basic inspection and descriptive analysis before training the classifier.

### Dataset snapshot

```text
cgpa   resume_score   placed
8.14      6.52          1
6.17      5.17          0
8.27      8.86          1
6.88      7.27          1
7.52      7.30          1
```

Because the dataset contains two input features, it is also well suited for visualizing the classifier's **decision regions**.

---

## 🗂️ Repository Structure

```text
perceptron-deep-learning/
│
├── 📓 Perceptron.ipynb   # Complete notebook: exploration, training & evaluation
├── 📄 placement.csv      # Dataset used for binary classification
└── 📘 README.md          # Project documentation
```

---

## 🧪 What the Notebook Covers

### 1. Environment and imports

The notebook uses a lightweight Python data-science stack:

- **NumPy** for numerical operations
- **Pandas** for loading and inspecting tabular data
- **Matplotlib** for plotting
- **Seaborn** for visual exploration
- **Scikit-learn** for the Perceptron model, train/test splitting, and evaluation metrics
- **mlxtend** for decision-region visualization

### 2. Data loading

The dataset is loaded from the repository with:

```python
import pandas as pd

df = pd.read_csv("placement.csv")
df.head()
```

### 3. Data inspection

The notebook checks the shape, columns, data types, descriptive statistics, and missing values before modeling.

### 4. Feature/target preparation

The input features are:

```python
X = df[["cgpa", "resume_score"]]
```

The target is:

```python
y = df["placed"]
```

### 5. Train/test split

The data is divided into training and test sets using `train_test_split` so that model performance can be evaluated on unseen examples.

### 6. Perceptron training

The notebook uses Scikit-learn's `Perceptron` implementation to train the classifier on the placement dataset.

### 7. Evaluation

The notebook imports and uses:

```python
accuracy_score
confusion_matrix
classification_report
```

These provide complementary views of model performance:

- **Accuracy** — overall proportion of correct predictions.
- **Confusion matrix** — counts of correct and incorrect predictions by class.
- **Classification report** — precision, recall, F1-score, and support.

### 8. Decision-region visualization

The notebook also uses `plot_decision_regions` from `mlxtend` to visualize how the trained Perceptron separates the two classes in feature space.

---

## 🚀 Getting Started

### Prerequisites

Install Python 3.x and a Jupyter-compatible environment.

### 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/perceptron-deep-learning.git
cd perceptron-deep-learning
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Perceptron.ipynb
```

Then execute the cells from top to bottom.

---

## 💻 Minimal Perceptron Example

A simplified version of the workflow looks like this:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("placement.csv")

# Features and target
X = df[["cgpa", "resume_score"]]
y = df["placed"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = Perceptron()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

> **Note:** The exact numeric results depend on the train/test split and model configuration used in the notebook.

---

## 🧠 Why the Perceptron Matters

The Perceptron is simple, but the idea behind it is foundational.

Modern neural networks are vastly more capable, yet they still build on familiar concepts:

```text
Perceptron
   ↓
Artificial Neuron
   ↓
Layers of Neurons
   ↓
Neural Networks
   ↓
Deep Neural Networks
```

Studying the Perceptron first makes later topics such as **logistic regression, multilayer perceptrons, backpropagation, activation functions, gradient-based optimization, and deep neural networks** much easier to understand.

---

## ✅ Strengths

- Easy to understand and implement.
- Fast for small linearly separable classification problems.
- Excellent starting point for learning neural-network fundamentals.
- Makes the idea of weights and bias concrete.
- Produces an interpretable linear decision boundary.

## ⚠️ Limitations

The classic Perceptron is a **linear classifier**. That means it cannot correctly represent arbitrary non-linearly separable patterns.

For problems requiring richer representations, the natural next step is to move from a single linear neuron to **multilayer neural networks** and learn how backpropagation enables them to model complex relationships.

---

## 🔬 Key Concepts to Study Next

This repository is a strong starting point for a deep-learning progression:

```text
Perceptron
   ↓
Activation Functions
   ↓
Neural Network Architecture
   ↓
Forward Propagation
   ↓
Loss Functions
   ↓
Gradient Descent
   ↓
Backpropagation
   ↓
MLP / ANN
   ↓
CNN → Computer Vision
   ↓
RNN / LSTM → Sequential Data
   ↓
Transformers → Modern NLP & Multimodal AI
```

---

## 📚 Recommended Study Questions

After completing the notebook, make sure you can answer these without looking at the code:

- What problem does a Perceptron solve?
- Why do we need weights?
- What is the purpose of the bias term?
- What makes a decision boundary linear?
- What happens when a sample is misclassified?
- Why can a single Perceptron not solve XOR?
- What is the difference between a Perceptron and a multilayer neural network?
- How are accuracy, precision, recall, and F1-score different?

These questions matter more than simply memorizing the API.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **NumPy** | Numerical computing |
| **Pandas** | Dataset loading and analysis |
| **Matplotlib** | Visualization |
| **Seaborn** | Statistical visualization |
| **Scikit-learn** | Perceptron model and evaluation utilities |
| **mlxtend** | Decision-region plotting |
| **Jupyter Notebook** | Interactive experimentation and learning |

---

## 🌱 Project Philosophy

This repository follows a simple rule:

> **Learn the mechanism before hiding behind the framework.**

The point of this project is not to build the most sophisticated classifier. It is to develop the intuition needed to understand what a neural model is actually doing under the hood.

That foundation pays off later. A lot.

---

## 🔮 Future Extensions

Possible improvements for this learning project include:

- Implement the Perceptron **manually using NumPy**, without Scikit-learn.
- Visualize the weight updates over training iterations.
- Plot the decision boundary before and after training.
- Experiment with different learning rates and iteration counts.
- Demonstrate the Perceptron's failure on the XOR problem.
- Compare the Perceptron with Logistic Regression.
- Build a simple multilayer neural network from scratch.
- Add backpropagation and gradient descent step by step.

---

## 🤝 Contributing

This is primarily a learning repository, but improvements are welcome.

A useful contribution should ideally make the project **clearer, more reproducible, or more educational**.

```text
Fork → Improve → Commit → Pull Request
```

---

## 👤 Author

**Maganpreet Singh**

Learning and building in **Machine Learning & Deep Learning**, one concept at a time.

GitHub: [@Maganpreet-Singh](https://github.com/Maganpreet-Singh)

---

## ⭐ Support the Journey

If this repository helps you understand the fundamentals of neural networks, consider giving it a ⭐.

More importantly: **keep building.**

---

<div align="center">

### 🧠 Understand the neuron. Build the network. Learn the system.

</div>
