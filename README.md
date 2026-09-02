<div align="center">

# 🧠 Perceptron — Deep Learning From Scratch

### Building intuition for neural networks, one neuron at a time.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/mlxtend-Visualization-2E7D32?style=for-the-badge" alt="mlxtend">
</p>

<p>
  <strong>Understand the Perceptron → understand the foundations of neural networks.</strong>
</p>

</div>

---

## 📌 About This Project

This repository is a **fundamentals-first Deep Learning project** focused on the **Perceptron**, one of the simplest and most important models for binary classification.

The project uses a small student-placement dataset to explore how a linear classifier learns from input features such as **CGPA** and **resume score**. The notebook walks through data loading, exploration, preprocessing, model training, evaluation, and visualization.

The goal is not simply to call a machine-learning API. The goal is to understand what is happening behind the API:

```text
Features
   ↓
Weighted Sum
   ↓
Bias
   ↓
Activation / Decision
   ↓
Prediction
   ↓
Error
   ↓
Parameter Update
   ↓
Learning
```

> 💡 A Perceptron is small. The idea behind it is not.

---

## 🎯 Learning Objectives

By completing this project, you should be able to explain:

- What an artificial neuron is.
- Why a model needs **weights** and **bias**.
- How a Perceptron creates a binary prediction.
- What a **linear decision boundary** means.
- How supervised learning uses labelled examples.
- How classification models are evaluated.
- Why a single Perceptron cannot solve every classification problem.
- How the Perceptron connects to modern neural networks.

This makes the repository a useful starting point before moving into **ANNs, MLPs, backpropagation, CNNs, RNNs, and Transformers**.

---

## 🧮 The Mathematics Behind a Perceptron

For an input vector:

$$
x = [x_1, x_2, ..., x_n]
$$

with weights:

$$
w = [w_1, w_2, ..., w_n]
$$

and bias $b$, the Perceptron calculates:

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

or more compactly:

$$
z = w^Tx + b$$

A threshold-style decision function then produces the predicted class:

$$
\hat{y} = f(z)
$$

For a binary classifier, the decision can be represented conceptually as:

```text
if z >= 0:
    predict class 1
else:
    predict class 0
```

### 🧠 Why this matters

This tiny equation introduces several concepts that appear repeatedly in Deep Learning:

**parameters → weighted combinations → activation → prediction → error → learning**

Once these ideas become intuitive, deeper neural-network architectures stop feeling like magic.

---

## 🔁 Perceptron Learning Intuition

The Perceptron learns by repeatedly examining training examples and adjusting its parameters when the current prediction is incorrect.

A simplified update rule can be expressed as:

$$
w \leftarrow w + \eta(y - \hat{y})x$$

$$
b \leftarrow b + \eta(y - \hat{y})$$

where:

| Symbol | Meaning |
|---|---|
| $w$ | Weight vector |
| $b$ | Bias |
| $x$ | Input features |
| $y$ | True class |
| $\hat{y}$ | Predicted class |
| $\eta$ | Learning rate |

The important idea is simple:

> **Wrong prediction → adjust parameters → try again.**

---

## 📊 Dataset

The repository contains `placement.csv`, a compact binary-classification dataset with **100 samples** and three columns. The notebook confirms the dataset shape as `(100, 3)`.

| Column | Type | Role |
|---|---|---|
| `cgpa` | Float | Student academic feature |
| `resume_score` | Float | Resume-related feature |
| `placed` | Integer | Binary target: `0` or `1` |

Example records from the dataset:

| `cgpa` | `resume_score` | `placed` |
|---:|---:|---:|
| 8.14 | 6.52 | 1 |
| 6.17 | 5.17 | 0 |
| 8.27 | 8.86 | 1 |
| 6.88 | 7.27 | 1 |
| 7.52 | 7.30 | 1 |

The two numerical input features also make the dataset suitable for visualizing a two-dimensional decision boundary.

---

## 🔍 Exploratory Data Analysis

The notebook performs basic inspection before model training, including:

- Dataset shape
- Column names
- Data types
- Descriptive statistics
- Missing-value checks
- Feature/target separation
- Visual exploration

The observed dataset contains two numerical features and a binary target, making it a clean example for learning linear classification.

---

## 🗂️ Repository Structure

```text
perceptron-deep-learning/
│
├── 📓 Perceptron.ipynb
│   └── Main notebook: data exploration, model training,
│       evaluation, and decision-region visualization
│
├── 📓 Perceptron Code.ipynb
│   └── Additional Perceptron-focused notebook
│
├── 📓 Perceptron Loss Function.ipynb
│   └── Notebook focused on the Perceptron loss function
│
├── 📄 placement.csv
│   └── Dataset used for binary classification
│
└── 📘 README.md
    └── Project documentation
```

---

## 🧪 Main Notebook Workflow

### 1. Import the required libraries

The main notebook uses:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from mlxtend.plotting import plot_decision_regions
```

### 2. Load the dataset

```python
df = pd.read_csv("placement.csv")
df.head()
```

### 3. Inspect the data

The notebook checks:

```python
df.shape
df.columns
df.info()
df.describe()
```

### 4. Select features and target

```python
X = df[["cgpa", "resume_score"]]
y = df["placed"]
```

### 5. Split the dataset

The data is divided into training and testing sets so the trained model can be evaluated on unseen examples.

### 6. Train the Perceptron

```python
model = Perceptron()
model.fit(X_train, y_train)
```

### 7. Generate predictions

```python
y_pred = model.predict(X_test)
```

### 8. Evaluate the classifier

The notebook uses three complementary evaluation tools:

```python
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```

### 9. Visualize decision regions

The project uses `plot_decision_regions` from `mlxtend` to make the learned classification boundary easier to understand.

---

## 📈 Model Evaluation

### Accuracy

Accuracy measures the proportion of predictions that are correct:

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

### Confusion Matrix

A confusion matrix summarizes classification results by showing counts of:

- True Positives
- True Negatives
- False Positives
- False Negatives

### Classification Report

The classification report provides:

| Metric | Meaning |
|---|---|
| Precision | How many predicted positives were actually positive |
| Recall | How many actual positives were correctly detected |
| F1-score | Harmonic mean of precision and recall |
| Support | Number of true samples in each class |

> The repository focuses on learning the workflow rather than claiming that a small educational dataset represents a production-grade placement predictor.

---

## 🚀 Getting Started

### Prerequisites

Install:

- Python 3.x
- Jupyter Notebook or JupyterLab

### Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/perceptron-deep-learning.git
cd perceptron-deep-learning
```

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend jupyter
```

### Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
Perceptron.ipynb
```

Run the notebook cells from top to bottom.

---

## 💻 Minimal Working Example

Here is the core workflow in one place:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("placement.csv")

# Features and target
X = df[["cgpa", "resume_score"]]
y = df["placed"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Perceptron
model = Perceptron()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

> **Note:** Exact metric values depend on the train/test split and model configuration used when the notebook is executed.

---

## 🧠 Why the Perceptron Matters

The Perceptron is historically simple, but conceptually important.

A useful mental model is:

```text
                    ┌─────────────┐
Input Features ───► │   Weights   │
                    └──────┬──────┘
                           │
                           ▼
                    Weighted Sum
                           │
                         + Bias
                           │
                           ▼
                     Activation
                           │
                           ▼
                       Prediction
```

A single Perceptron produces a **linear decision boundary**. That makes it powerful enough for some problems but fundamentally limited for problems that require nonlinear separation.

The classic example is **XOR**: a single linear Perceptron cannot represent the XOR decision pattern.

That limitation is exactly why learning progresses naturally from:

```text
Single Perceptron
        ↓
Multiple Neurons
        ↓
Multilayer Perceptron (MLP)
        ↓
Forward Propagation
        ↓
Loss Functions
        ↓
Gradient Descent
        ↓
Backpropagation
        ↓
Deep Neural Networks
```

---

## ✅ Strengths

- Simple and fast.
- Excellent for learning binary classification.
- Makes weights and bias tangible.
- Produces an interpretable linear decision boundary.
- Provides a strong conceptual bridge into neural networks.

## ⚠️ Limitations

- It is a **linear classifier**.
- It cannot model arbitrary nonlinear decision boundaries.
- It is not suitable as a general-purpose deep-learning architecture.
- Real-world predictive systems require better data, validation, feature design, model selection, and domain-specific evaluation.

In other words: **the Perceptron is a foundation, not the final destination.**

---

## 🔬 Hands-On Experiments to Try

Turn this repository into a stronger learning lab by experimenting with:

### Experiment 1 — Change the train/test split

Compare different test sizes and observe how evaluation changes.

### Experiment 2 — Change model hyperparameters

Experiment with parameters such as:

- `max_iter`
- `eta0`
- `tol`
- `random_state`

### Experiment 3 — Visualize the decision boundary

Compare how the learned boundary changes after retraining.

### Experiment 4 — Test linear separability

Create a synthetic dataset where classes are clearly separable and compare it with a non-linearly separable dataset.

### Experiment 5 — Break the model with XOR

Use XOR to see a core limitation of a single linear neuron.

### Experiment 6 — Compare algorithms

Compare the Perceptron with:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine

This is where the difference between **linear models** and **nonlinear decision-making** becomes much clearer.

---

## 🧭 Learning Roadmap After This Project

This repository fits naturally into a broader Deep Learning journey:

```text
✅ Classical Machine Learning
        ↓
✅ Perceptron
        ↓
🔜 Activation Functions
        ↓
🔜 Artificial Neural Networks
        ↓
🔜 Forward Propagation
        ↓
🔜 Loss Functions
        ↓
🔜 Gradient Descent
        ↓
🔜 Backpropagation
        ↓
🔜 Optimizers
        ↓
🔜 Regularization
        ↓
🔜 CNNs — Computer Vision
        ↓
🔜 RNN / LSTM — Sequential Data
        ↓
🔜 Attention
        ↓
🔜 Transformers
        ↓
🔜 Modern Deep Learning Applications
```

The fastest way forward is not to memorize every architecture. **Build small models, inspect the math, make mistakes, debug them, and repeat.**

---

## 📚 Questions You Should Be Able to Answer

After finishing the project, try answering these without opening the notebook:

1. What is a Perceptron?
2. What is the role of a weight?
3. What is the role of bias?
4. Why is the decision boundary linear?
5. How does a Perceptron update its parameters?
6. What does a learning rate control?
7. What is the difference between a prediction and a true label?
8. What does a confusion matrix tell you?
9. Why can a single Perceptron not solve XOR?
10. What changes when we move from one neuron to a multilayer network?

If you can answer these clearly, you are not just using a model—you are starting to understand one.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Programming language |
| **NumPy** | Numerical operations |
| **Pandas** | Data loading and analysis |
| **Matplotlib** | Plotting and visualization |
| **Seaborn** | Statistical visualization |
| **Scikit-learn** | Perceptron model and evaluation |
| **mlxtend** | Decision-region visualization |
| **Jupyter Notebook** | Interactive experimentation |

---

## 🌱 Project Philosophy

> **Learn the mechanism before hiding behind the framework.**

High-level libraries are useful. Understanding the underlying idea is better.

The purpose of this repository is to build the kind of intuition that makes later topics easier to reason about—not to pretend that one small notebook is a production ML system.

Start with one neuron. Understand it deeply. Then stack the neurons.

---

## 🔮 Future Improvements

Planned learning extensions include:

- [ ] Implement a Perceptron **from scratch using NumPy**.
- [ ] Implement the Perceptron loss manually.
- [ ] Visualize weight updates over iterations.
- [ ] Plot the decision boundary before and after training.
- [ ] Demonstrate XOR failure.
- [ ] Compare Perceptron vs Logistic Regression.
- [ ] Build a small neural network from scratch.
- [ ] Implement gradient descent manually.
- [ ] Implement backpropagation step by step.
- [ ] Move from a single neuron to an MLP / ANN.

---

## 🤝 Contributing

This repository is primarily a learning project, and educational improvements are welcome.

Good contributions should improve one or more of the following:

- Clarity
- Reproducibility
- Mathematical understanding
- Code quality
- Visual explanations
- Learning value

Typical workflow:

```text
Fork → Improve → Commit → Pull Request
```

---

## 👤 Author

### Maganpreet Singh

Machine Learning & Deep Learning learner building projects to understand models from fundamentals to practical applications.

**GitHub:** [@Maganpreet-Singh](https://github.com/Maganpreet-Singh)

---

## ⭐ Support the Journey

If this repository helps you learn something, consider giving it a ⭐ and exploring the other machine-learning projects in the profile.

The real goal is bigger than one repository:

> **Learn. Build. Break. Debug. Repeat.**

---

<div align="center">

### 🧠 One neuron today. Deep learning tomorrow.

</div>
