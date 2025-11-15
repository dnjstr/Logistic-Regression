import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegressionProb:
    def __init__(self, lr=0.0001, epochs=2000, threshold=0.5):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.loss_history = []
        self.acc_history = []

    def compute_loss(self, y, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def accuracy(self, y, y_pred_labels):
        return np.mean(y == y_pred_labels)

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            # gradients
            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # track loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # track accuracy
            y_pred_labels = (y_pred >= self.threshold).astype(int)
            acc = self.accuracy(y, y_pred_labels)
            self.acc_history.append(acc)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        p1 = sigmoid(linear_model)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T  # return [[p0, p1], ...]

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


# Example dataset
X = np.array([[150],[160],[170],[180],[190],[200]])
X = (X - X.mean()) / X.std()
y = np.array([0,0,0,1,1,1])

model = LogisticRegressionProb(lr=0.01, epochs=300)
model.fit(X, y)

probs = model.predict_proba(X)
pred = model.predict(X)

print("Predicted Probabilities:", np.round(probs, 2).tolist())
print("Predicted Labels:", pred.tolist())
print("Actual:", y.tolist())

# Plot loss
model.plot_loss()
