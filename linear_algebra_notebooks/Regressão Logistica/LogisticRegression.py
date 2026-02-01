import time
import matplotlib.pyplot as plt
from find_minima import GradientDescent, NewtonRaphson
import numpy as np
def timer(func):
    def wrapper(*args, **kwargs):
        if kwargs.get('markTime', False):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Execution time: {end - start:.6f} seconds")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper
   

class LogisticRegression:
    def __init__(self, X, Y, b0, learning_rate=0.001):
        self.X = X
        self.Y = Y 
        self.alfa = learning_rate
        self.beta = b0

        self.methods = {"Gradient Descent": GradientDescent,
                        "NewtonRaphson": NewtonRaphson}

    def inv_Logistic_link(self, theta):
        # Use np.clip to avoid overflow in exp for large negative/positive theta
        theta = np.clip(theta, -500, 500)
        return 1 / (1 + np.exp(-theta))
    def grad(self, theta):
        predictions = self.inv_Logistic_link(self.X @ theta)
        
        # Calcula class weights
        n_samples = len(self.Y)
        n_classes = 2
        n_class_0 = np.sum(self.Y == 0)
        n_class_1 = np.sum(self.Y == 1)
        
        # Weight inversamente proporcional à frequência
        weight_0 = n_samples / (n_classes * n_class_0)
        weight_1 = n_samples / (n_classes * n_class_1)
        
        # Aplica pesos aos erros
        weights = np.where(self.Y == 1, weight_1, weight_0)
        errors = (predictions - self.Y) * weights
        
        return self.X.T @ errors

    @timer
    def optimize(self, method="Gradient Descent", markTime=False, maxIterations=1000):
        return self.methods[method](self.alfa,self.Y, self.X, self.grad, self.beta.copy(), maxIterations=maxIterations)
    def plot_losses(self, losses):
        """Plot the loss over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Loss', color='blue')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.legend()
        plt.grid()
        plt.show()
    def plot_loss_surface_zoomed(self):
        """Plot loss surface zoomed around the optimum"""
        

        if len(self.beta) != 2:
            print(f"Warning: plotting only first 2 of {len(self.beta)} dimensions")
        
        center1, center2 = self.beta[0], self.beta[1]
        
        # Varia ±5 ao redor do ótimo (não ±40!)
        beta1_range = np.linspace(center1 - 1, center1 + 1, 50)
        beta2_range = np.linspace(center2 - 1, center2 + 1, 50)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        losses = []
        for b1 in beta1_range:
            for b2 in beta2_range:
                beta_temp = self.beta.copy()  # ← mantém outros betas!
                beta_temp[0] = b1
                beta_temp[1] = b2
                
                predictions = self.inv_Logistic_link(self.X @ beta_temp)
                loss = -np.mean(self.Y * np.log(predictions + 1e-15) + 
                            (1 - self.Y) * np.log(1 - predictions + 1e-15))
                losses.append((b1, b2, loss))
        
        losses = np.array(losses)
        X_grid = losses[:, 0].reshape(len(beta1_range), len(beta2_range))
        Y_grid = losses[:, 1].reshape(len(beta1_range), len(beta2_range))
        Z_grid = losses[:, 2].reshape(len(beta1_range), len(beta2_range))
        
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Beta 1')
        ax.set_ylabel('Beta 2')
        ax.set_zlabel('Custo')
        ax.set_title('Superfície Côncava')
        ax.view_init(elev=30, azim=120)
        ax.legend()
        plt.show()
            
    @classmethod
    def Logistic_link(cls, theta):
        
        epsilon = 1e-15
        theta = np.clip(theta, epsilon, 1 - epsilon)
        return np.log(theta / (1 - theta))
    def calculate_scores(self, X, Y, b):
        '''
        returns a dict of accuracy, precision, recall, f1_score for each class
        '''
        classes = {0:{},1:{}}
        predictions = self.inv_Logistic_link(X @ b)
        predicted_classes = (predictions >= 0.5).astype(int)
        print(predicted_classes)
        for actual, predicted in zip(Y, predicted_classes):
            classes[actual]["predicted"] = predicted
            accuracy = np.mean(predicted_classes == Y)
            recall = np.sum((predicted_classes == 1) & (Y == actual)) / np.sum(Y == actual) if np.sum(Y == actual) > 0 else 0
            precision = np.sum((predicted_classes == actual) & (Y == actual)) /sum(predicted_classes == actual) if np.sum(predicted_classes == actual) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall)

            classes[actual]["accuracy"] = accuracy
            classes[actual]["precision"] = precision
            classes[actual]["recall"] = recall
            classes[actual]["f1_score"] = f1_score
        
     
        return  classes
