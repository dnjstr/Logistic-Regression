import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
        self.losses = []

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

            # compute and store loss
            loss = - (1/self.m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
            self.losses.append(loss)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        df = df.dropna()

        if 'cp' in df.columns and df['cp'].dtype == object:
            df = pd.get_dummies(df, columns=['cp'], drop_first=True)

        return df

    def split_and_scale_data(self, df, test_size=0.2):
        """Split data and scale features"""
        X = df.drop(columns=['target'])
        y = df['target'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()