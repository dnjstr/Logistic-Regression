import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function with stability
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000, threshold=0.5, scale_features=True):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.scale_features = scale_features
        self.loss_history = []
        self.acc_history = []

    # -----------------------------------
    # Automatic Feature Scaling
    # -----------------------------------
    def _scale(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # avoid division by zero
        self.std[self.std == 0] = 1  

        return (X - self.mean) / self.std

    # -----------------------------------
    # Loss Function
    # -----------------------------------
    def _loss(self, y, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # -----------------------------------
    # Accuracy Tracker
    # -----------------------------------
    def _accuracy(self, y, y_pred_labels):
        return np.mean(y == y_pred_labels)

    # -----------------------------------
    # Fit Model
    # -----------------------------------
    def fit(self, X, y):
        # Scale features
        if self.scale_features:
            X = self._scale(X)

        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            # Gradients
            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss
            loss = self._loss(y, y_pred)
            self.loss_history.append(loss)

            # Track accuracy
            y_pred_labels = (y_pred >= self.threshold).astype(int)
            acc = self._accuracy(y, y_pred_labels)
            self.acc_history.append(acc)

    # -----------------------------------
    # Predict Probabilities
    # -----------------------------------
    def predict_proba(self, X):
        if self.scale_features:
            X = (X - self.mean) / self.std

        linear_model = np.dot(X, self.weights) + self.bias
        p1 = sigmoid(linear_model)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T

    # -----------------------------------
    # Predict Class Labels
    # -----------------------------------
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]  # p(class1)
        return (probs >= self.threshold).astype(int)

    # -----------------------------------
    # Plot Loss Curve
    # -----------------------------------
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


# ---------------------------------------------------
# Example dataset (Hours studied vs Pass/Fail)
# ---------------------------------------------------
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X, y)

probs = model.predict_proba(X)
preds = model.predict(X)

print("Predicted Probabilities:", np.round(probs, 3).tolist())
print("Predicted Labels:", preds.tolist())
print("Actual:", y.tolist())

# Plot loss curve
model.plot_loss()
