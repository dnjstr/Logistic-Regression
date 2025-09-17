import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # only for splitting
from sklearn.preprocessing import StandardScaler      # only for scaling
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("Working directory:", os.getcwd())

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []   # store losses
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # gradients
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # compute and store loss (binary cross-entropy)
            loss = - (1/self.m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
            self.losses.append(loss)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

# Load the dataset
df = pd.read_csv("Real Datasets/heart_disease.csv")  # adjust if needed

# Example preprocessing
df = df.dropna()

# Encode categorical variables if needed
if df['cp'].dtype == object:
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target'].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model_cls = LogisticRegressionScratch(lr=0.01, epochs=5000)
model_cls.fit(X_train_scaled, y_train)
y_pred = model_cls.predict(X_test_scaled)
probas = model_cls.predict_proba(X_test_scaled)

# Evaluate accuracy
accuracy = np.mean(y_pred == y_test)
print("Classification Accuracy:", accuracy)

# Show some predictions
for i in range(5):
    print(f"Sample {i}: true={y_test[i]}, prob={probas[i]:.4f}, pred_label={y_pred[i]}")

# ----------------------------
# Visualization
# ----------------------------

# 1) Training Loss Curve
plt.figure(figsize=(8,5))
plt.plot(model_cls.losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss (Binary Cross-Entropy)")
plt.title("Logistic Regression Training Convergence")
plt.legend()
plt.show()

# 2) Confusion Matrix with Labels
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Heart Disease", "Heart Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression Scratch")
plt.show()

# 3) ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probas)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression Scratch")
plt.legend()
plt.show()