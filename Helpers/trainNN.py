import numpy as np
import matplotlib.pyplot as plt
from .plottingNN import plot_network
import time
import sys
import os

# Correction de l'importation problématique
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Helpers import Helpers

def compute_accuracy(neural_network, data_set, mode='train'):
    """
    Compute the accuracy of the neural network on a given dataset.
    
    Args:
        neural_network: The neural network to evaluate
        data_set: The dataset to use for evaluation
        mode: 'train' or 'val' to use training or validation set
    
    Returns:
        accuracy: Classification accuracy (0.0-100.0)
    """
    # Save original testing phase state
    original_phase = neural_network.phase
    neural_network.phase = True  # Set to testing phase
    
    # Get the appropriate data based on mode
    if mode == 'train':
        inputs, labels = data_set.get_train_set()
    elif mode == 'val':
        inputs, labels = data_set.get_validation_set()
        if inputs is None or labels is None:
            print("No validation set available. Skipping validation accuracy.")
            return None
    
    # Check for empty inputs
    if inputs.size == 0 or labels.size == 0:
        print(f"Warning: Empty {mode} set detected. Skipping accuracy calculation.")
        return None
    
    # For large datasets, use a subset for faster computation
    max_samples = 1000  # Maximum number of samples to use for accuracy computation
    if inputs.shape[0] > max_samples:
        indices = np.random.choice(inputs.shape[0], max_samples, replace=False)
        inputs = inputs[indices]
        labels = labels[indices]
    
    try:
        # Forward pass through the network
        predictions = neural_network.test(inputs)
        
        # Calculate accuracy
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes) * 100  # Scale to 0-100%
    except Exception as e:
        print(f"Error computing {mode} accuracy: {str(e)}")
        accuracy = None
    
    # Restore original phase
    neural_network.phase = original_phase
    
    return accuracy


def compute_regression_metrics(neural_network, data_set, mode='train'):
    """
    Compute regression metrics for the neural network on a given dataset.
    
    Args:
        neural_network: The neural network to evaluate
        data_set: The dataset to use for evaluation
        mode: 'train' or 'val' to use training or validation set
    
    Returns:
        metrics: Dictionary with regression metrics (mse, rmse, mae, r_squared, explained_variance)
    """
    # Save original testing phase state
    original_phase = neural_network.phase
    neural_network.phase = True  # Set to testing phase
    
    # Get the appropriate data based on mode
    if mode == 'train':
        inputs, labels = data_set.get_train_set()
    elif mode == 'val':
        inputs, labels = data_set.get_validation_set()
        if inputs is None or labels is None:
            print(f"No validation set available. Skipping {mode} metrics.")
            return None
    
    # Check for empty inputs
    if inputs.size == 0 or labels.size == 0:
        print(f"Warning: Empty {mode} set detected. Skipping metrics calculation.")
        return None
    
    # For large datasets, use a subset for faster computation
    max_samples = 1000  # Maximum number of samples to use for metrics computation
    if inputs.shape[0] > max_samples:
        indices = np.random.choice(inputs.shape[0], max_samples, replace=False)
        inputs = inputs[indices]
        labels = labels[indices]
    
    try:
        # Forward pass through the network
        predictions = neural_network.test(inputs)
        
        # Calculate regression metrics
        metrics = Helpers.calculate_regression_metrics(predictions, labels)
    except Exception as e:
        print(f"Error computing {mode} metrics: {str(e)}")
        metrics = None
    
    # Restore original phase
    neural_network.phase = original_phase
    
    return metrics


def train_network(neural_network, iterations, metrics_interval=10, plot_interval=None):
    """
    Train the neural network for a specified number of iterations
    with simple output of loss and metrics.
    
    Args:
        neural_network: The neural network to train
        iterations: Number of training iterations
        metrics_interval: Interval for computing and displaying metrics
        plot_interval: If provided, plot the network every plot_interval iterations
        
    Returns:
        dict: A dictionary containing training history (loss, metrics)
    """
    neural_network.phase = False  # Set to training phase
    
    # Détection automatique du type de problème en fonction de la couche de perte
    loss_layer_class = neural_network.loss_layer.__class__.__name__
    
    # Liste des classes de perte pour la régression
    regression_loss_classes = ['MSELoss', 'L2Loss', 'L1Loss', 'MAELoss', 'HuberLoss', 'RegressionLoss']
    
    # Déterminer le type de problème
    problem_type = 'regression' if loss_layer_class in regression_loss_classes else 'classification'
    print(f"Détection automatique du type de problème: {problem_type} (basé sur {loss_layer_class})")
    
    # Store metrics history
    history = {
        'loss': [],
        'iterations': []
    }
    
    # Add appropriate metrics to history based on problem type
    if problem_type == 'classification':
        history.update({
            'train_accuracy': [],
            'val_accuracy': []
        })
    else:  # regression
        history.update({
            'train_metrics': [],
            'val_metrics': []
        })
    
    # Check if validation set exists - improved version
    has_validation = False
    try:
        if hasattr(neural_network.data_layer, 'get_validation_set'):
            # Call get_validation_set only once to avoid unnecessary computation
            val_inputs, val_labels = neural_network.data_layer.get_validation_set()
            
            # Proper validation check
            if (val_inputs is not None and val_labels is not None and 
                val_inputs.size > 0 and val_labels.size > 0):
                has_validation = True
                print("Validation set detected and will be used for display.")
            else:
                print("Validation set method exists but returned empty data.")
    except Exception as e:
        print(f"Error checking validation set: {str(e)}")
        has_validation = False
    
    # Record start time
    start_time = time.time()
    
    # Print header for metrics based on problem type
    if problem_type == 'classification':
        print(f"{'Iteration':>10} | {'Loss':>12} | {'Train Acc (%)':>14}", end="")
        if has_validation:
            print(f" | {'Val Acc (%)':>12}", end="")
    else:  # regression
        print(f"{'Iteration':>10} | {'Loss':>12} | {'Train MSE':>14}", end="")
        if has_validation:
            print(f" | {'Val MSE':>12}", end="")
    
    print(" | Elapsed Time")
    print("-" * (50 + (15 if has_validation else 0)))
    
    for i in range(iterations):
        # Forward and backward pass
        try:
            loss = neural_network.forward()
            neural_network.loss.append(loss)
            neural_network.backward()
            
            # Store loss value
            history['loss'].append(loss)
        except Exception as e:
            print(f"Error in iteration {i}: {str(e)}")
            # Use previous loss if available, otherwise use a placeholder
            if history['loss']:
                history['loss'].append(history['loss'][-1])
            else:
                print("No previous loss available. Using 0.0 as placeholder.")
                history['loss'].append(0.0)
            continue
        
        # Compute and store metrics at specified intervals
        if i % metrics_interval == 0 or i == iterations - 1:
            elapsed_time = time.time() - start_time
            history['iterations'].append(i)
            
            # Compute metrics based on problem type
            if problem_type == 'classification':
                # Compute training accuracy
                train_acc = compute_accuracy(neural_network, neural_network.data_layer, mode='train')
                if train_acc is not None:
                    history['train_accuracy'].append(train_acc)
                
                # Compute validation accuracy if available
                val_acc = None
                if has_validation:
                    try:
                        val_acc = compute_accuracy(neural_network, neural_network.data_layer, mode='val')
                        if val_acc is not None:
                            history['val_accuracy'].append(val_acc)
                    except Exception as e:
                        print(f"Error computing validation accuracy: {str(e)}")
                
                # Print metrics
                print(f"{i:10d} | {loss:12.6f} | ", end="")
                if train_acc is not None:
                    print(f"{train_acc:14.2f}", end="")
                else:
                    print(f"{'N/A':>14}", end="")
                    
                if has_validation:
                    if val_acc is not None:
                        print(f" | {val_acc:12.2f}", end="")
                    else:
                        print(f" | {'N/A':>12}", end="")
            
            else:  # regression
                # Compute training metrics
                train_metrics = compute_regression_metrics(neural_network, neural_network.data_layer, mode='train')
                if train_metrics is not None:
                    history['train_metrics'].append(train_metrics)
                
                # Compute validation metrics if available
                val_metrics = None
                if has_validation:
                    try:
                        val_metrics = compute_regression_metrics(neural_network, neural_network.data_layer, mode='val')
                        if val_metrics is not None:
                            history['val_metrics'].append(val_metrics)
                    except Exception as e:
                        print(f"Error computing validation metrics: {str(e)}")
                
                # Print metrics
                print(f"{i:10d} | {loss:12.6f} | ", end="")
                if train_metrics is not None:
                    print(f"{train_metrics['mse']:14.6f}", end="")
                else:
                    print(f"{'N/A':>14}", end="")
                    
                if has_validation:
                    if val_metrics is not None:
                        print(f" | {val_metrics['mse']:12.6f}", end="")
                    else:
                        print(f" | {'N/A':>12}", end="")
            
            print(f" | {elapsed_time:.2f}s")
        
        # Plot the network at specified intervals if requested
        if plot_interval is not None and i > 0 and i % plot_interval == 0:
            try:
                plot_network(neural_network, title=f"Network at Iteration {i}", display=True)
            except Exception as e:
                print(f"Error plotting network: {str(e)}")
    
    # Print completion message
    total_time = time.time() - start_time
    print("-" * (50 + (15 if has_validation else 0)))
    print(f"Training completed in {total_time:.2f}s")
    
    # Create a final summary plot if we have data
    if len(history['loss']) > 0:
        plt.figure(figsize=(12, 8))
        
        # Create two subplots
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        
        # Plot loss in the first subplot
        ax1.plot(range(len(history['loss'])), history['loss'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot metrics in the second subplot based on problem type
        if problem_type == 'classification' and history['iterations'] and 'train_accuracy' in history and history['train_accuracy']:
            # Interpolate training accuracy for plotting
            all_iterations = list(range(len(history['loss'])))
            interpolated_train_acc = np.interp(all_iterations,
                                              history['iterations'],
                                              history['train_accuracy'])
            
            # Plot accuracies in the second subplot
            ax2.plot(all_iterations, interpolated_train_acc, 'g-', label='Training Accuracy')
            
            if has_validation and 'val_accuracy' in history and history['val_accuracy']:
                interpolated_val_acc = np.interp(all_iterations,
                                                history['iterations'],
                                                history['val_accuracy'])
                ax2.plot(all_iterations, interpolated_val_acc, 'r-', label='Validation Accuracy')
            
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_ylim(0, 105)  # Accuracy from 0-100% with a small margin
        
        elif problem_type == 'regression' and history['iterations'] and 'train_metrics' in history and history['train_metrics']:
            # Extract MSE values for training
            train_mse = [metrics['mse'] for metrics in history['train_metrics']]
            
            # Interpolate training MSE for plotting
            all_iterations = list(range(len(history['loss'])))
            interpolated_train_mse = np.interp(all_iterations,
                                              history['iterations'],
                                              train_mse)
            
            # Plot MSE in the second subplot
            ax2.plot(all_iterations, interpolated_train_mse, 'g-', label='Training MSE')
            
            if has_validation and 'val_metrics' in history and history['val_metrics']:
                # Extract MSE values for validation
                val_mse = [metrics['mse'] for metrics in history['val_metrics']]
                
                interpolated_val_mse = np.interp(all_iterations,
                                               history['iterations'],
                                               val_mse)
                ax2.plot(all_iterations, interpolated_val_mse, 'r-', label='Validation MSE')
            
            ax2.set_title('Training and Validation MSE')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Mean Squared Error')
        
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return history