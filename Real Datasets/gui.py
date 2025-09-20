import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import ttk, messagebox
import seaborn as sns

# Import our custom logistic regression module
from logistic_regression import LogisticRegressionScratch, DataProcessor, sigmoid

plt.style.use('default')
sns.set_palette("husl")

print("Working directory:", os.getcwd())

class LogisticRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction - Logistic Regression Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.data_processor = DataProcessor()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.probas = None
        self.accuracy = None
        self.df = None
        self.feature_names = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(main_frame, text="Model Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Parameters
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, padx=(0, 5))
        self.lr_var = tk.DoubleVar(value=0.01)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=2, padx=(0, 5))
        self.epochs_var = tk.IntVar(value=5000)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(params_frame, text="Test Size:").grid(row=0, column=4, padx=(0, 5))
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(params_frame, textvariable=self.test_size_var, width=10).grid(row=0, column=5)
        
        # Buttons Frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, pady=(0, 10))
        
        # Buttons
        ttk.Button(buttons_frame, text="Load Data", command=self.load_data).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(buttons_frame, text="Train Model", command=self.train_model).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(buttons_frame, text="Show Graphs", command=self.show_graphs).grid(row=0, column=2)
        
        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=4, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Plots area with cross borders
        self.plots_area = tk.Frame(results_frame, bg='white', relief='sunken', bd=2)
        self.plots_area.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure equal weights for rows and columns to make perfect squares
        self.plots_area.grid_rowconfigure(0, weight=1, uniform="row")
        self.plots_area.grid_rowconfigure(1, weight=1, uniform="row")
        self.plots_area.grid_columnconfigure(0, weight=1, uniform="col")
        self.plots_area.grid_columnconfigure(1, weight=1, uniform="col")
        
        # Create 4 quadrants with equal borders
        self.quad1 = tk.Frame(self.plots_area, bg='lightgray', relief='solid', bd=1)
        self.quad1.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 1), pady=(0, 1))
        
        self.quad2 = tk.Frame(self.plots_area, bg='lightgray', relief='solid', bd=1)
        self.quad2.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(1, 0), pady=(0, 1))
        
        self.quad3 = tk.Frame(self.plots_area, bg='lightgray', relief='solid', bd=1)
        self.quad3.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 1), pady=(1, 0))
        
        self.quad4 = tk.Frame(self.plots_area, bg='lightgray', relief='solid', bd=1)
        self.quad4.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(1, 0), pady=(1, 0))
    
    # for loading the dataset
    def load_data(self):
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Loading dataset...\n")
            self.root.update()
            
            self.df = self.data_processor.load_and_preprocess_data("Real Datasets/heart_disease.csv")
            
            self.results_text.insert(tk.END, f"Data loaded successfully! Shape: {self.df.shape}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    # For training the model
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Training model...\n")
            self.root.update()
            
            # Split and scale data
            self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test, self.feature_names = \
                self.data_processor.split_and_scale_data(self.df, self.test_size_var.get())
            
            # Train model
            self.model = LogisticRegressionScratch(lr=self.lr_var.get(), epochs=self.epochs_var.get())
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            self.y_pred = self.model.predict(self.X_test_scaled)
            self.probas = self.model.predict_proba(self.X_test_scaled)
            self.accuracy = np.mean(self.y_pred == self.y_test)
            
            # Show results
            results = f"Training completed!\n"
            results += f"Accuracy: {self.accuracy:.4f}\n"
            results += f"Parameters: LR={self.lr_var.get()}, Epochs={self.epochs_var.get()}\n"
            results += "Ready to show graphs!\n"
            
            self.results_text.insert(tk.END, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    # For showing the graphs
    def show_graphs(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train model first!")
            return
        
        try:
            for widget in self.quad1.winfo_children():
                widget.destroy()
            for widget in self.quad2.winfo_children():
                widget.destroy()
            for widget in self.quad3.winfo_children():
                widget.destroy()
            for widget in self.quad4.winfo_children():
                widget.destroy()
            
            self.create_training_loss_plot()
            self.create_confusion_matrix_plot()
            self.create_sigmoid_plot()  # Changed from ROC to Sigmoid
            self.create_feature_weights_plot()
            
            # To show the update results
            self.results_text.delete(1.0, tk.END)
            results = f"Graphs displayed!\n"
            results += f"Accuracy: {self.accuracy:.4f}\n"
            results += f"Sample predictions:\n"
            for i in range(min(3, len(self.y_test))):
                results += f"  Sample {i}: True={self.y_test[i]}, Pred={self.y_pred[i]}, Prob={self.probas[i]:.3f}\n"
            
            self.results_text.insert(tk.END, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Plot error: {str(e)}")
    
    # for training loss (top-left)
    def create_training_loss_plot(self):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(self.model.losses, color='blue', linewidth=2)
        ax.set_title('Training Loss', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Epochs', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        plt.tight_layout(pad=2.0)
        
        canvas = FigureCanvasTkAgg(fig, self.quad1)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # for Sigmoid function (bottom-left)
    def create_sigmoid_plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Generate values for the generic sigmoid function
        z = np.linspace(-10, 10, 200)
        s = sigmoid(z)
        ax.plot(z, s, color='blue', linewidth=2, label="Sigmoid Curve")
    
    # Overlay actual model predictions on test set
        if self.probas is not None:
        # Use first feature for visualization (project to 1D)
            X_test_1d = self.X_test_scaled[:, 0]  
            sort_idx = np.argsort(X_test_1d)
            ax.scatter(X_test_1d, self.probas, 
                   c=self.y_test, cmap="bwr", edgecolor="k", alpha=0.7, 
                   label="Predicted Probabilities")
        
        # Fit a line based on model weights for visualization
            linear_z = np.linspace(X_test_1d.min(), X_test_1d.max(), 200)
            linear_probs = sigmoid(linear_z * self.model.weights[0] + self.model.bias)
            ax.plot(linear_z, linear_probs, color="green", linewidth=2, linestyle="--", label="Model Sigmoid Fit")
    
    # Add decision threshold
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label="Threshold = 0.5")
    
    # Labels & legend
        ax.set_title('Sigmoid Function & Model Predictions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Input (z)')
        ax.set_ylabel('σ(z)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    
        plt.tight_layout(pad=2.0)
    
        canvas = FigureCanvasTkAgg(fig, self.quad3)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    
    # For confusion matrix (top-right)
    def create_confusion_matrix_plot(self):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))  # Match other plots' size
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=["No Heart Disease", "Heart Disease"]
        )
        disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format='d')
        
        ax.set_title('Confusion Matrix - Logistic Regression Scratch', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Predicted label', fontsize=9)
        ax.set_ylabel('True label', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        for text in ax.texts:
            text.set_fontweight('bold')
            text.set_fontsize(12)
        
        plt.subplots_adjust(left=0.25, right=0.95, top=0.30, bottom=0.12)
        plt.tight_layout(pad=1.5)
        
        canvas = FigureCanvasTkAgg(fig, self.quad2)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # For feature weights (bottom-right)
    def create_feature_weights_plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        
        feature_cols = ['age', 'sex', 'chol']
        
        feature_descriptions = {
            'age': 'age in years',
            'sex': '(1 = male; 0 = female)',
            'chol': 'serum cholestoral in mg/dl'
        }
        
        for i, feature in enumerate(feature_cols):
            ax = axes[i]
            
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                ax.hist(data, bins=15, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
                
                ax.set_title(f'# {feature}', fontsize=12, fontweight='bold', pad=10)
                
                if feature in feature_descriptions:
                    ax.text(0.5, 0.95, feature_descriptions[feature], transform=ax.transAxes,
                           ha='center', va='top', fontsize=9, style='italic')
                
                ax.grid(True, alpha=0.3)
                ax.set_ylabel('Frequency')
                
                stats_text = f'Valid: {len(data)}\nMean: {data.mean():.1f}\nStd. Dev: {data.std():.1f}'
                ax.text(0.98, 0.85, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), 
                       fontsize=9)
                
                ax.text(0.02, 0.02, f'{data.min():.0f}', transform=ax.transAxes, 
                       fontsize=10, ha='left', va='bottom', fontweight='bold')
                ax.text(0.98, 0.02, f'{data.max():.0f}', transform=ax.transAxes, 
                       fontsize=10, ha='right', va='bottom', fontweight='bold')
                
            else:
                ax.text(0.5, 0.5, f'Feature "{feature}"\nnot found', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
                ax.set_title(f'# {feature}', fontsize=12, fontweight='bold')
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.85, wspace=0.3)  # <-- Add wspace for more horizontal space
        
        canvas = FigureCanvasTkAgg(fig, self.quad4)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticRegressionGUI(root)
    root.mainloop()