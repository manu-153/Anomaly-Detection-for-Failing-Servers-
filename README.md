# 🚨 Anomaly Detection using Gaussian Distribution

## 📌 Project Overview

This project implements an anomaly detection algorithm using Gaussian distribution to detect failing servers in a network. The core objective is to model the distribution of normal behavior and identify data points that significantly deviate from this distribution as anomalies.

We apply the approach to:

1. A simple 2-feature dataset (latency, throughput)
2. A high-dimensional dataset (11 features)

---

## 📊 Dataset Description

The data is loaded using utility functions:

* `X_train`: Training data
* `X_val`, `y_val`: Cross-validation data and ground truth labels
* `X_train_high`, `X_val_high`, `y_val_high`: High-dimensional dataset

---

## 🧠 Methodology

### 1. **Estimate Gaussian Parameters**

We compute the **mean (μ)** and **variance (σ²)** for each feature in the dataset.

```python
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = (1 / m) * np.sum((X - mu) ** 2, axis=0)
    return mu, var
```

---

### 2. **Compute Probabilities**

We evaluate the probability of each data point using the **Multivariate Gaussian distribution**.

```python
p = multivariate_gaussian(X, mu, var)
```

---

### 3. **Select Threshold for Anomaly Detection**

Using the cross-validation set, we choose an optimal threshold (`epsilon`) that maximizes the **F1 score**.

```python
def select_threshold(y_val, p_val):
    # Iteratively find the best epsilon based on F1 score
```

---

### 4. **Detect and Visualize Anomalies**

Data points with probability less than `epsilon` are flagged as anomalies. These are visualized using scatter plots with red circles around anomalous points.

---

## 📈 Visualizations

* Scatter plots of training data
* Contours representing the learned Gaussian distribution
* Red circles around detected anomalies

---

## 🧪 Results

### 🟢 2D Dataset:

* Best epsilon: `8.99e-05`
* Best F1 Score: `0.875`
* Anomalies Found: Visualized on the plot

### 🔵 High-Dimensional Dataset:

* Best epsilon: `1.38e-18`
* Best F1 Score: `0.615`
* Anomalies Found: `117`

---

## 📦 File Structure

```
.
├── anomaly_detection.py      # Main script
├── utils.py                  # Helper functions: load_data(), multivariate_gaussian(), visualize_fit()
├── README.md                 # Project documentation
└── data/                     # Optional: Store datasets here
```

---

## ✅ Dependencies

* Python 3.x
* NumPy
* Matplotlib

Install with:

```bash
pip install numpy matplotlib
```

---

## 📚 References

* Andrew Ng's Machine Learning course (Coursera)
* Multivariate Gaussian distribution theory

---

## 🧠 Future Work

* Apply to time-series or streaming data
* Use PCA for dimensionality reduction
* Try other models like Isolation Forest, One-Class SVM

