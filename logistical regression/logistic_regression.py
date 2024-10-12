import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=40):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []
        
    def sigmoid_func(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid(self, X):
        return np.array([self.sigmoid_func(z) for z in X])
    
    def compute_loss(self, y, y_predictive):
        loss = -y.T.dot(np.log(self.sigmoid(y_predictive))) + ((1-y).T.dot(np.log(1-self.sigmoid(y_predictive))))
        return np.mean(loss)
    
    def update_weights(self,X, y):  # Make predictions
        n = X.shape[0] # n: #datapoint
        z = np.dot(X, self.weights) + self.bias 
        y_predicted = self.sigmoid(z)
        dw = np.dot(X.T, (y_predicted - y))/n
        db = np.sum(y_predicted - y)/n 
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

        return y_predicted 
    
    def fit(self, X, y):
        n = X.shape[0] # Number of data points
        m = X.shape[1] # m: #features
        self.weights= np.zeros(m)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = self.update_weights(X, y)  
            current_loss = self.compute_loss(y, y_predicted)  
            self.losses.append(current_loss)
            y_threshold = [1 if _y > 0.5 else 0 for _y in y_predicted]
            current_accuracy = self.accuracy(y, y_threshold)
            self.train_accuracies.append(current_accuracy)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(z)
        return [1 if _y > 0.5 else 0 for _y in y_predicted]
    
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)

