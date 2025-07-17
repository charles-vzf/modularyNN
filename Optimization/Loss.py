import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        # Store predictions and labels for backward pass
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        
        loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon)) # a float value
        return loss

    def backward(self, label_tensor):
        error_tensor = -label_tensor / (self.prediction_tensor + self.epsilon)
        return error_tensor # shape (batch_size, num_classes)


class L2Loss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


class L1Loss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.abs(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return np.sign(np.subtract(self.input_tensor, label_tensor))

class HingeLoss:
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.sum(np.maximum(0, 1 - input_tensor * label_tensor))

    def backward(self, label_tensor):
        return np.where(label_tensor * self.input_tensor < 1, -label_tensor, 0)

class SoftmaxLoss:
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        exp_input = np.exp(input_tensor - np.max(input_tensor))
        softmax_output = exp_input / np.sum(exp_input)
        return -np.sum(label_tensor * np.log(softmax_output + 1e-10))

    def backward(self, label_tensor):
        exp_input = np.exp(self.input_tensor - np.max(self.input_tensor))
        softmax_output = exp_input / np.sum(exp_input)
        return softmax_output - label_tensor

class RegressionLoss:
    """Classe de base pour les fonctions de perte de régression."""
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None
    
    def compute_metrics(self, prediction_tensor, label_tensor):
        """
        Calcule diverses métriques pour évaluer les performances de régression.
        
        Returns:
        - dict: Dictionnaire contenant les métriques (MSE, MAE, R²)
        """
        # Calcul du MSE
        mse = np.mean(np.square(prediction_tensor - label_tensor))
        
        # Calcul du MAE
        mae = np.mean(np.abs(prediction_tensor - label_tensor))
        
        # Calcul du R² (coefficient de détermination)
        ss_total = np.sum(np.square(label_tensor - np.mean(label_tensor)))
        ss_residual = np.sum(np.square(label_tensor - prediction_tensor))
        if ss_total == 0:  # Éviter division par zéro
            r_squared = 0
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        # Calcul du RMSE
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'r_squared': r_squared,
            'rmse': rmse
        }

# MSELoss existe déjà, hériter de RegressionLoss
class MSELoss(RegressionLoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.mean(np.square(input_tensor - label_tensor))
    
    def backward(self, label_tensor):
        return 2 * (self.input_tensor - label_tensor) / label_tensor.size

# Ajout de MAELossReg
class MAELoss(RegressionLoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.mean(np.abs(input_tensor - label_tensor))
    
    def backward(self, label_tensor):
        return np.sign(self.input_tensor - label_tensor) / label_tensor.size

# Ajout de HuberLoss (robuste aux outliers)
class HuberLoss(RegressionLoss):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        
        error = input_tensor - label_tensor
        abs_error = np.abs(error)
        
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * np.square(quadratic) + self.delta * linear
        return np.mean(loss)
    
    def backward(self, label_tensor):
        error = self.input_tensor - label_tensor
        abs_error = np.abs(error)
        
        # Gradient de la perte de Huber
        gradient = np.where(abs_error <= self.delta, 
                            error, 
                            self.delta * np.sign(error))
        
        return gradient / label_tensor.size



class ExponentialLoss:
    """Exponential loss function, often used in boosting algorithms."""
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.sum(np.exp(-label_tensor * input_tensor))

    def backward(self, label_tensor):
        return -label_tensor * np.exp(-label_tensor * self.input_tensor)




class LogisticLoss:
    """Logistic loss function, commonly used in binary classification."""
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.sum(np.log(1 + np.exp(-label_tensor * input_tensor)))

    def backward(self, label_tensor):
        return -label_tensor / (1 + np.exp(label_tensor * self.input_tensor))




class SquaredHingeLoss:
    """Squared hinge loss function, often used in SVMs."""
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        return np.sum(np.square(np.maximum(0, 1 - label_tensor * input_tensor)))

    def backward(self, label_tensor):
        return -2 * label_tensor * np.where(label_tensor * self.input_tensor < 1, 1, 0)