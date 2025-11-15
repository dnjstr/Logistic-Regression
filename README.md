# **Logistic Regression From Scratch – GUI Application**

A Python project that implements **Logistic Regression from scratch** (using only NumPy) and provides an **interactive Tkinter GUI** for training, evaluating, and visualizing a classification model—specifically designed for **Heart Disease Prediction**.

## **📌 Project Overview**

This project demonstrates:

### ✔ Logistic Regression implemented manually

* Gradient Descent optimizer
* Sigmoid activation
* Binary Cross-Entropy loss
* Custom accuracy, precision, recall calculations

### ✔ Full Tkinter-based GUI

Users can:

* Load the Heart Disease dataset
* Adjust model hyperparameters
* Train the model
* Visualize multiple graphs simultaneously

### ✔ Visual Output (4-Quadrant Graphing Panel)

The GUI displays:

1. **Training Loss Curve**
2. **Confusion Matrix**
3. **Sigmoid Curve + Model Predictions**
4. **Feature Distribution Histograms**


## **📁 Project Structure**

```
project/
│
├── logistic_regression.py    # Logistic Regression and Data Processor classes
├── gui.py                    # Tkinter GUI application
├── Real Datasets/
│     └── heart_disease.csv   # Dataset used by the program
│
└── README.md                 # Documentation
```

# **📦 Requirements**

Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tk
```

> *Note: Tkinter is already included in most Python installations.*


# **▶ Running the Program**

### **1. Ensure the dataset exists**

Place your `heart_disease.csv` inside:

```
Real Datasets/heart_disease.csv
```

### **2. Run the GUI**

```bash
python gui.py
```

The application window will open automatically.


# **🧠 Core Components**

## **1. LogisticRegressionScratch**

A fully manual implementation of Logistic Regression.

### Features:

* Weight initialization
* Gradient descent updates
* Sigmoid activation
* Probability prediction
* Binary class prediction
* Loss tracking for plotting

### Key methods:

* `fit(X, y)`
* `predict(X)`
* `predict_proba(X)`



## **2. DataProcessor**

Handles:

* CSV loading
* Dropping missing values
* One-hot encoding (`cp` column handling)
* Train-test splitting
* Feature scaling via `StandardScaler`


## **3. Evaluation Metrics**

Custom function `calculate_metrics()` computes:

* Accuracy
* Precision
* Recall


# **🖥 GUI Features**

### **✔ Load Data**

Loads and preprocesses the Heart Disease dataset:

* Cleans missing values
* Encodes categories
* Displays dataset size

### **✔ Train Model**

Uses values provided by user:

* **Learning Rate**
* **Number of Epochs**
* **Test Size**

Displays:

* Accuracy
* First few predictions with probabilities

### **✔ Show Graphs**

Displays 4 graphs arranged in a 2×2 layout:

#### **1️⃣ Training Loss Plot**

Shows how the model converges over epochs.

#### **2️⃣ Confusion Matrix**

Displays model performance on test data.

#### **3️⃣ Sigmoid Function & Predictions**

* Plots the sigmoid curve
* Overlays predicted probabilities
* Shows decision boundary

#### **4️⃣ Feature Distribution Histograms**

For features:

* `age`
* `sex`
* `chol`


# **📊 Example Output**

Upon training, the text output shows:

```
Training completed!
Accuracy: 0.8420
Parameters: LR=0.01, Epochs=5000
Ready to show graphs!
```

Sample predictions:

```
Sample 0: True=1, Pred=1, Prob=0.892
Sample 1: True=0, Pred=0, Prob=0.104
Sample 2: True=1, Pred=1, Prob=0.743
```


# **📌 Notes**

* The model is purely **from scratch**—no scikit-learn logistic regression is used.
* GUI elements resize dynamically for large screens (1400×900 by default).
* Graphs are embedded using `FigureCanvasTkAgg`.


# **📄 License**

This project is free to use, modify, and distribute for educational and academic purposes.
