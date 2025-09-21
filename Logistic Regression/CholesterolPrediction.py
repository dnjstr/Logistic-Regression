import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

class LogisticRegressionProb:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
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

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

# Example dataset (Cholesterol levels vs Disease probability)/ DUMMY DATASETS
X = np.array([[150],[160],[170],[180],[190],[200]])
y = np.array([0,0,0,1,1,1])

# ONLY CHANGE: Reduce learning rate and epochs to prevent over-convergence
model = LogisticRegressionProb(lr=0.001, epochs=500)
model.fit(X, y)
probs = model.predict_proba(X)

print("Predicted Probabilities:", probs)
print("Actual:", y.tolist())