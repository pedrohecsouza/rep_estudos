import numpy as np
tol = 1e-9
def NewtonRaphson(alfa,Y,X, grad_func, b, verbose=True, maxIterations=1000):
    losses = []
    for i in range(maxIterations):
        g = grad_func(b)
 
      
        g_temp = g.copy()
        if verbose and i % 10 == 0:
            predictions = 1 / (1 + np.exp(-X @ b))
            loss = -np.mean(Y * np.log(predictions + 1e-15) + 
                           (1 - Y) * np.log(1 - predictions + 1e-15))
            losses.append(loss)
        
        if np.linalg.norm(g) < tol :
            print(f"Converged at iteration {i}")
            return b
        
        # Compute Hessian matrix
        predictions = 1 / (1 + np.exp(-X @ b))
        W = np.diag(predictions * (1 - predictions))
        H = X.T @ W @ X
        
        # Update weights using Newton-Raphson formula
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Hessian is singular, cannot invert. Stopping.")
            return b
        
        b -= alfa * H_inv @ g
        

    return [b, losses]
def GradientDescent(alfa,Y,X, grad_func, b, verbose=True, maxIterations=1000):
    losses = []
    for i in range(maxIterations):
        g = grad_func(b)
 
        
        g_temp = g.copy()
        if verbose and i % 10 == 0:
            predictions = 1 / (1 + np.exp(-X @ b))
            loss = -np.mean(Y * np.log(predictions + 1e-15) + 
                           (1 - Y) * np.log(1 - predictions + 1e-15))
            losses.append(loss)
        
        if np.linalg.norm(g) < tol :
            print(f"Converged at iteration {i}")
            return b, losses
        
        # Add gradient clipping to avoid exploding weights
        grad_norm = np.linalg.norm(g)
        max_norm =1
        if grad_norm > max_norm:
            g = g * (max_norm / grad_norm)
        
        b -= alfa * g
        normalized_gradient = g / np.linalg.norm(g)
    
    return b, losses
