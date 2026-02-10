"""
linear_regression

Production-ready implementation using gradient descent optimization
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression with Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent (α).
    iterations : int, default=1000
        Number of gradient descent iterations.
    verbose : bool, default=False
        Print progress during training.
    
    Attributes
    ----------
    weights : ndarray
        Learned coefficients.
    bias : float
        Learned intercept.
    cost_history : list
        Cost at each iteration.
    """

    def __init__(self, learning_rate=0.01, iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _compute_cost(self, X, y, y_pred):
        """
        Implements and computes Mean Squared Error cost.
        
        J(θ) = (1/2m) Σ(ŷ - y)²
        
        Parameters
        ----------
        X : ndarray
            Features.
        y : ndarray
            True values.
        y_pred : ndarray
            Predicted values.
            
        Returns
        -------
        cost : float
        """
        m = len(y)
        return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    
    def fit(self, X, y):
        """
        Fit model using gradient descent.
        
        Parameters
        ----------
        X : ndarray of shape (m, n)
            Training features.
        y : ndarray of shape (m,)
            Target values.
        
        Returns
        -------
        self : object
        """
        m, n = X.shape
        
        # Initialize
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Cost
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
            
            # Gradients
            error = y_pred - y
            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Verbose
            if self.verbose and (i % 100 == 0 or i == 0):
                print(f"Iteration {i:4d}: Cost = {cost:,.2f}")
        
        if self.verbose:
            print(f"Final cost: {cost:,.2f}")
        
        return self
    
    def predict(self, X):
        """
        Predict using linear model.
        
        Parameters
        ----------
        X : ndarray
            Samples.
        
        Returns
        -------
        y_pred : ndarray
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        R² = 1 - (SS_residual / SS_total)
        
        Parameters
        ----------
        X : ndarray
            Test samples.
        y : ndarray
            True values.
        
        Returns
        -------
        r2 : float
        """
        y_pred = self.predict(X)
        ss_residual = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def plot_learning_curve(self, figsize=(10, 6), save_path=None):
        """Visualize training progress."""
        if not self.cost_history:
            raise ValueError("No history. Train model first.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.cost_history, linewidth=2, color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Stats
        initial = self.cost_history[0]
        final = self.cost_history[-1]
        reduction = ((initial - final) / initial) * 100
        
        print(f"\nTraining Summary:")
        print(f"  Initial cost: {initial:,.2f}")
        print(f"  Final cost: {final:,.2f}")
        print(f"  Reduction: {reduction:.1f}%")
