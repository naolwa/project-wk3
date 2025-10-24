
# 🧠 Machine Learning and Deep Learning Model Demo (Streamlit + PyTorch + Scikit-learn)

## 📘 Overview

This project demonstrates two key machine learning tasks:

1. **Task 2 – Decision Tree Classification on the Iris Dataset**
2. **Task 3 – Convolutional Neural Network (CNN) for MNIST Handwritten Digit Recognition**

Both tasks are integrated into a **Streamlit web app** that allows users to interactively explore how traditional ML and deep learning models work.

---

## 🚀 Features

### 🌸 Task 2: Decision Tree Classifier (Iris Dataset)

* Uses **Scikit-learn’s Iris dataset** for flower classification.
* Allows users to **select which features** to train on via Streamlit’s interface.
* Handles **missing values automatically** using mean imputation.
* Trains a **Decision Tree Classifier** and evaluates it using:

  * Accuracy
  * Precision
  * Recall
* Displays the dataset and selected features in an interactive web UI.

### 🌸 Task 3: Convolutional Neural Network (CNN)

* Implements a **CNN in PyTorch** to classify handwritten digits from the **MNIST dataset**.
* Performs preprocessing with normalization and tensor conversion.
* Uses:

  * 2 convolutional layers
  * 1 fully connected hidden layer
  * ReLU activation
  * Dropout for regularization
* Trains the CNN for multiple epochs and evaluates its accuracy on the test dataset.
* Visualizes predictions for a few sample images using **Matplotlib**.
* Designed to run on either **CPU or GPU** (CUDA if available).

---

## 🧩 Project Structure

```
📦 ML_DL_Demo
 ┣ 📜 app.py                  # Main Streamlit + PyTorch + Sklearn code
 ┣ 📁 data                    # MNIST dataset (auto-downloaded)
 ┣ 📜 requirements.txt        # Dependencies
 ┣ 📜 README.md               # Project documentation
 ┗ 📜 images/                 # Optional folder for screenshots
```

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ML_DL_Demo.git
cd ML_DL_Demo
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate   # (on macOS/Linux)
myenv\Scripts\activate      # (on Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
numpy
pandas
scikit-learn
streamlit
torch
torchvision
matplotlib
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

This will launch a local Streamlit web app in your browser.

---

## 📊 Example Outputs

### 🌼 Decision Tree (Iris)

* **Metrics Example:**

  ```
  Accuracy:  0.95
  Precision: 0.94
  Recall:    0.95
  ```
* Interactive feature selection from Iris attributes:

  * Sepal length
  * Sepal width
  * Petal length
  * Petal width

### 🔢 CNN (MNIST)

* **Training Output Example:**

  ```
  Epoch 1/5, Loss: 0.2134
  Epoch 2/5, Loss: 0.1245
  ...
  Test Accuracy: 98.75%
  ```
* **Sample Prediction Visualization:**
  Displays 5 handwritten digits with their predicted labels.

---

## 🎯 Learning Objectives

* Understand how **traditional ML (Decision Trees)** differ from **Deep Learning (CNNs)**.
* Explore **data preprocessing** (imputation, normalization, encoding).
* Learn how to **evaluate model performance** using key metrics.
* Integrate machine learning models with **Streamlit** for interactive visualization.

---

## ⚖️ Bias & Ethical Considerations

### Potential Biases

| Model                | Bias Type       | Description                                           |
| -------------------- | --------------- | ----------------------------------------------------- |
| Decision Tree (Iris) | Dataset Bias    | Small dataset with limited diversity.                 |
| CNN (MNIST)          | Dataset Bias    | MNIST only includes neat, centered digits.            |
| Both                 | Evaluation Bias | Accuracy alone doesn’t capture class-specific errors. |

### Mitigation Strategies

* Use **TensorFlow Fairness Indicators** for per-class performance breakdowns.
* Apply **data augmentation** for more diversity.
* Use **balanced datasets** or **class weighting** during training.

---

## 🐛 Troubleshooting

Common issues and fixes:

| Problem                              | Cause                      | Solution                                                      |
| ------------------------------------ | -------------------------- | ------------------------------------------------------------- |
| `RuntimeError: Expected 4D input`    | Missing channel dimension  | Add `.unsqueeze(1)` or `[..., None]` to images                |
| `ValueError: Shapes not aligned`     | Wrong input size for `fc1` | Verify flattened tensor size before defining `Linear()` layer |
| `ImportError: No module named torch` | Missing PyTorch            | Run `pip install torch torchvision`                           |
| Streamlit doesn’t display plots      | Not using `st.pyplot()`    | Uncomment `st.pyplot()` after visualization                   |



---

Would you like me to create a **ready-to-download README.md file** (in proper Markdown format) and a **requirements.txt** file to match your code?
