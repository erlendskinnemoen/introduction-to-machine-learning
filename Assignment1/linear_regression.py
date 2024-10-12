import numpy as np

def accuracy(true_values, predictions):
    return np.mean(true_values == predictions)

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])  # Initialize weights to zero (x.shape[1] = number of features)
        self.bias = 0  # Initialize bias to zero
    
        n = len(y)
        # Gradient Descent
        for _ in range(self.epochs):
            y_pred = np.matmul(X, self.weights) + self.bias
            
            grad_w = (1 / n)  * np.dot(X.T, (y_pred - y))
            grad_b = (1 / n)  * np.sum(y_pred - y)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            loss = (1 /(2 * n) ) * np.sum((y_pred - y) ** 2) #MSE error for linear regression
            self.losses.append(loss)
            
            #pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            #self.train_accuracies.append(self.accuracy(y, pred_to_class))
    
    def predict(self, X):
        return np.matmul(X, self.weights) + self.bias

