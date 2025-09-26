import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Sigmoid activation function
def sigmoid(z):
    """
    Computes the sigmoid function for a given input z.
    The sigmoid function maps any real-valued number to a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))

# Function to calculate evaluation metrics (Accuracy, Precision, Recall)
def calculate_metrics(y_true, y_pred):
    """
    Calculates accuracy, precision, and recall for a binary classification model.

    Args:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Convert inputs to numpy arrays for reliable calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate True Positives, False Positives, False Negatives, and True Negatives
    # A positive class is typically labeled as 1.
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / len(y_true)

    # Calculate precision (avoiding division by zero)
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # Calculate recall (avoiding division by zero)
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

class LogisticRegressionScratch:
    """
    A Logistic Regression model implemented from scratch using gradient descent.
    """
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.
        """
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            # Forward pass: Calculate the predicted probabilities
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # Calculate gradients (derivatives of the loss function)
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            # Update weights and bias using gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute and store the loss (Binary Cross-Entropy)
            # A small epsilon is added to log to prevent log(0) errors
            loss = - (1/self.m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
            self.losses.append(loss)
    
    def predict_proba(self, X):
        """
        Predicts the probability of a sample belonging to the positive class.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predicts the class label (0 or 1) based on a threshold.
        """
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

class DataProcessor:
    """
    Handles data loading, preprocessing, splitting, and scaling.
    Modified to work with a DataFrame and return feature names.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset from a CSV file.
        Note: This function assumes the file is in the same directory.
        """
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        # Handle the 'cp' column which can be categorical
        if 'cp' in df.columns and df['cp'].dtype == 'object':
            df = pd.get_dummies(df, columns=['cp'], drop_first=True)
        
        return df
    
    def split_and_scale_data(self, df, test_size=0.2):
        """
        Splits data into training and testing sets, scales features,
        and returns the feature names for plotting.
        """
        X = df.drop(columns=['target'])
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Scale features using StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    # 1. Generate a synthetic dataset for demonstration
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=0, 
        random_state=42
    )

    # 2. Process and split the data
    # Create a dummy DataFrame to match the GUI's usage
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    processor = DataProcessor()
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = processor.split_and_scale_data(df)
    
    # 3. Initialize and train the Logistic Regression model
    model = LogisticRegressionScratch(lr=0.01, epochs=1000)
    model.fit(X_train_scaled, y_train)
    
    # 4. Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # 5. Evaluate the model using the custom metrics function
    metrics = calculate_metrics(y_test, y_pred)
    
    # 6. Print the results
    print("This is a module for a GUI. The code below demonstrates its functionality.")
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")