import numpy as np
import os
import random


# def enum_helpers():
#     '''
#     returns the list of available helpers (classes and functions)
#     '''
#     # TODO
#     return None


def gradient_check(layers, input_tensor, label_tensor, seed = None):
    '''
    Gradient check for the input tensor of the network.
    This function computes the numerical gradient of the loss function with respect to the input tensor
    and compares it to the analytical gradient computed by the network.
    '''
    epsilon = 1e-5
    difference = np.zeros_like(input_tensor)

    activation_tensor = input_tensor.copy()
    for layer in layers[:-1]:
        np.random.seed(seed) if seed is not None else None
        random.seed(seed) if seed is not None else None
        activation_tensor = layer.forward(activation_tensor)
    layers[-1].forward(activation_tensor, label_tensor)

    error_tensor = layers[-1].backward(label_tensor)
    for layer in reversed(layers[:-1]):
        error_tensor = layer.backward(error_tensor)

    it = np.nditer(input_tensor, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = input_tensor.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = input_tensor.copy()
        minus_epsilon[it.multi_index] -= epsilon

        analytical_derivative = error_tensor[it.multi_index]

        for layer in layers[:-1]:
            np.random.seed(seed) if seed is not None else None
            random.seed(seed) if seed is not None else None
            plus_epsilon = layer.forward(plus_epsilon)
            np.random.seed(seed) if seed is not None else None
            random.seed(seed) if seed is not None else None
            minus_epsilon = layer.forward(minus_epsilon)
        upper_error = layers[-1].forward(plus_epsilon, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        # print('Analytical: ' + str(analytical_derivative) + ' vs Numerical :' + str(numerical_derivative))
        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant

        it.iternext()
    return difference


def gradient_check_weights(layers, input_tensor, label_tensor, bias):
    '''
    Gradient check for the weights of the network.
    This function computes the numerical gradient of the loss function with respect to the weights
    and compares it to the analytical gradient computed by the network.
    '''
    epsilon = 1e-5
    if bias:
        weights = layers[0].bias
    else:
        weights = layers[0].weights
    difference = np.zeros_like(weights)

    it = np.nditer(weights, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = weights.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = weights.copy()
        minus_epsilon[it.multi_index] -= epsilon

        activation_tensor = input_tensor.copy()
        if bias:
            layers[0].bias = weights
        else:
            layers[0].weights = weights
        for layer in layers[:-1]:
            activation_tensor = layer.forward(activation_tensor)
        layers[-1].forward(activation_tensor, label_tensor)

        error_tensor = layers[-1].backward(label_tensor)
        for layer in reversed(layers[:-1]):
            error_tensor = layer.backward(error_tensor)
        if bias:
            analytical_derivative = layers[0].gradient_bias
        else:
            analytical_derivative = layers[0].gradient_weights
        analytical_derivative = analytical_derivative[it.multi_index]

        if bias:
            layers[0].bias = plus_epsilon
        else:
            layers[0].weights = plus_epsilon
        plus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            plus_epsilon_activation = layer.forward(plus_epsilon_activation)

        if bias:
            layers[0].bias = minus_epsilon
        else:
            layers[0].weights = minus_epsilon
        minus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            minus_epsilon_activation = layer.forward(minus_epsilon_activation)

        upper_error = layers[-1].forward(plus_epsilon_activation, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon_activation, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant

        it.iternext()
    return difference


def compute_bn_gradients(error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
    '''
    This function computes the gradients of the batch normalization layer with respect to the input tensor, weights, mean and variance.
    The gradients are computed using the chain rule and the backpropagation algorithm.
    '''
    # computation of the gradient w.r.t the input for the batch_normalization layer

    if eps > 1e-10:
        raise ArithmeticError("Eps must be lower than 1e-10. Your eps values %s" %(str(eps)))

    norm_mean = input_tensor - mean
    var_eps = var + eps

    gamma_err = error_tensor * weights
    inv_batch = 1. / error_tensor.shape[0]

    grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

    sqrt_var = np.sqrt(var_eps)
    first = gamma_err * 1. / sqrt_var

    grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=0)) * inv_batch
    grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=0)

    second = grad_var * (2. * norm_mean) * inv_batch
    grad_mu = grad_mu_two + grad_mu_one

    return first + second + inv_batch * grad_mu


def calculate_accuracy(results, labels):
    '''
    This function computes the accuracy of the network by comparing the predicted labels with the true labels.
    The predicted labels are obtained by taking the index of the maximum value in the results tensor.
    '''
    index_maximum = np.argmax(results, axis=1)
    one_hot_vector = np.zeros_like(results)
    for i in range(one_hot_vector.shape[0]):
        one_hot_vector[i, index_maximum[i]] = 1

    correct = 0.
    wrong = 0.
    for column_results, column_labels in zip(one_hot_vector, labels):
        if column_results[column_labels > 0.].all() > 0.:
            correct += 1.
        else:
            wrong += 1.

    return correct / (correct + wrong)


def calculate_regression_metrics(predictions, targets):
    """
    Calcule les métriques standard pour l'évaluation de la régression.
    
    Args:
        predictions: Tensor des valeurs prédites (batch_size, output_dim)
        targets: Tensor des valeurs cibles (batch_size, output_dim)
        
    Returns:
        dict: Dictionnaire contenant différentes métriques
    """
    # Mean Squared Error
    mse = np.mean(np.square(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # R² (coefficient de détermination)
    ss_total = np.sum(np.square(targets - np.mean(targets)))
    ss_residual = np.sum(np.square(targets - predictions))
    if ss_total == 0:  # Éviter division par zéro
        r_squared = 0
    else:
        r_squared = 1 - (ss_residual / ss_total)
    
    # Explained Variance Score
    var_pred = np.var(predictions)
    var_target = np.var(targets)
    if var_target == 0:  # Éviter division par zéro
        explained_variance = 0
    else:
        explained_variance = 1 - (np.var(targets - predictions) / var_target)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'explained_variance': explained_variance
    }