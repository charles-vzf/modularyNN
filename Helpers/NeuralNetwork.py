import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

# Use relative imports to avoid circular dependencies
from .plottingNN import plot_network
from .trainNN import train_network

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []              # List to store loss values after each training iteration
        self.layers = []
        self.data_layer = None      # Layer providing input data and labels
        self.loss_layer = None

    def help(self):
        """
        Checks if everything is correctly set up in the neural network and if not gives parts to initialize
        """
        if self.data_layer is None:
            print("Data layer is not set. Please set the data layer using 'self.data_layer = <data_layer>'")
            return False
        if self.loss_layer is None:
            print("Loss layer is not set. Please set the loss layer using 'self.loss_layer = <loss_layer>'")
            return False
        if len(self.layers) == 0:
            print("No layers added to the network. Please add layers using 'self.append_layer(<layer>)'")
            return False
        return True

    @property
    def phase(self):
        return all(layer.testing_phase for layer in self.layers)

    @phase.setter
    def phase(self, phase):
        for layer in self.layers:
            layer.testing_phase = phase
    
    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.current_label_tensor = label_tensor  # Store label for use in backward pass

        reg_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer.regularizer:
                reg_loss += layer.optimizer.regularizer.norm(layer.weights)

        data_loss = self.loss_layer.forward(input_tensor, self.current_label_tensor)
        total_loss = data_loss + reg_loss

        return total_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)

    def train(self, iterations, metrics_interval=10, plot_interval=None):
        """
        Train the neural network for a specified number of iterations
        
        Args:
            iterations (int): Number of training iterations
            metrics_interval (int): Interval for computing and displaying metrics
            plot_interval (int, optional): If provided, plot the network every plot_interval iterations
            
        Returns:
            dict: A dictionary containing training history (loss, metrics)
        """
        return train_network(self, iterations, metrics_interval, plot_interval)

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor
    
    def plot(self, title="Neural Network Architecture", detailed_params=True, display=True):
        """
        Visualize the neural network architecture with trainable and non-trainable parameters.
        
        Args:
            title (str): Title for the plot
            detailed_params (bool): Whether to display detailed parameter values for small networks
            display (bool): Whether to display the plot immediately
        """
        plot_network(self, title, detailed_params, display)

    @staticmethod
    def save(filename, net):
        """Save the neural network to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(net, f)

    @staticmethod
    def load(filename, data_layer):
        """Load the neural network from a file."""
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer  # Reassign the data layer after loading
        return net

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_layer'] = None  # Exclude the data layer from being pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data_layer = None  # Reinitialize the data layer to None