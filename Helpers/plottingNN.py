import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_network(neural_network, title="Neural Network Architecture", detailed_params=True, display=True):
    """
    Visualize the neural network architecture with trainable and non-trainable parameters.
    Uses a weighted graph approach to prevent overlapping when multiple trainable layers exist.
    
    Args:
        neural_network: The neural network object to plot
        title (str): Title for the plot
        detailed_params (bool): Whether to display detailed parameter values for small networks
        display (bool): Whether to display the plot immediately
    """
    # Count total trainable parameters
    total_params = 0
    for layer in neural_network.layers:
        if hasattr(layer, 'get_params_count'):
            total_params += layer.get_params_count()
        elif layer.trainable and hasattr(layer, 'weights'):
            total_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'bias'):
                total_params += np.prod(layer.bias.shape)
    
    # Determine if this is a small network that should show detailed visualization
    is_small_network = total_params < 200
    show_detailed = detailed_params and is_small_network
    
    # Create a directed graph with weighted edges
    G = nx.DiGraph()
    
    # Create figure with appropriate size and layout
    if show_detailed:
        figsize = (18, 12)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 1, height_ratios=[1, 2])
    else:
        figsize=(16, 8)
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
    
    ax_net = fig.add_subplot(gs[0])
    ax_params = fig.add_subplot(gs[1])
    
    # Add nodes to the graph with node attributes
    layer_labels = {}
    node_colors = {}
    node_sizes = {}
    node_types = {}  # Store the type of each node (input, trainable, non-trainable, output)
    
    # Input node
    G.add_node("Input")
    layer_labels["Input"] = "Input"
    node_colors["Input"] = 'lightblue'
    node_sizes["Input"] = 1000
    node_types["Input"] = "IO"
    
    # Organize layers by trainable status for proper vertical positioning
    trainable_indices = []
    non_trainable_indices = []
    
    for i, layer in enumerate(neural_network.layers):
        layer_name = f"Layer_{i}"
        
        # Create label based on layer type
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'get_layer_info'):
            layer_info = layer.get_layer_info()
            layer_labels[layer_name] = f"{layer_type}\n{layer_info}"
        elif layer_type == "Conv":
            shape_str = f"{layer.convolution_shape}→{layer.num_kernels}"
            layer_labels[layer_name] = f"Conv\n{shape_str}"
        elif layer_type == "FullyConnected":
            shape_str = f"{layer.input_dim}→{layer.output_dim}"
            layer_labels[layer_name] = f"FC\n{shape_str}"
        elif layer_type == "Pooling":
            layer_labels[layer_name] = f"Pool\n{layer.pooling_shape}"
        else:
            layer_labels[layer_name] = layer_type
        
        # Add node with appropriate attributes
        G.add_node(layer_name)
        
        # Choose node color and size based on layer type
        if layer.trainable:
            node_colors[layer_name] = 'lightgreen'
            node_sizes[layer_name] = 1200
            node_types[layer_name] = "Trainable"
            trainable_indices.append(i)
        else:
            node_colors[layer_name] = 'lightsalmon'
            node_sizes[layer_name] = 800
            node_types[layer_name] = "Non-trainable"
            non_trainable_indices.append(i)
    
    # Add output node
    G.add_node("Output")
    layer_labels["Output"] = "Output"
    node_colors["Output"] = 'lightblue'
    node_sizes["Output"] = 1000
    node_types["Output"] = "IO"
    
    # Add edges to the graph with weights reflecting parameter counts
    # Connect input to first layer
    G.add_edge("Input", f"Layer_0", weight=1)
    
    # Connect all layers
    for i in range(len(neural_network.layers) - 1):
        current_layer = neural_network.layers[i]
        next_layer = neural_network.layers[i+1]
        
        # Calculate edge weight based on parameters
        edge_weight = 1  # Default weight
        
        if current_layer.trainable and hasattr(current_layer, 'weights'):
            params_count = np.prod(current_layer.weights.shape)
            if hasattr(current_layer, 'bias'):
                params_count += np.prod(current_layer.bias.shape)
            # Normalize weight for visualization (log scale to prevent huge differences)
            edge_weight = max(1, np.log10(params_count + 1))
        
        G.add_edge(f"Layer_{i}", f"Layer_{i+1}", weight=edge_weight)
    
    # Connect last layer to output
    G.add_edge(f"Layer_{len(neural_network.layers)-1}", "Output", weight=1)
    
    # Calculate positions for the nodes
    # Use a multi-level approach for the y-coordinate:
    # - Input and Output nodes are at y=0
    # - Trainable layers are positioned above at different y values
    # - Non-trainable layers are positioned below at different y values
    positions = {}
    
    # Position Input and Output nodes
    positions["Input"] = (0, 0)
    positions["Output"] = (len(neural_network.layers) + 1, 0)
    
    # Number of trainable and non-trainable layers for spacing
    num_trainable = len(trainable_indices)
    num_non_trainable = len(non_trainable_indices)
    
    # Position trainable layers (above the center line)
    if num_trainable > 0:
        vertical_spacing = 1.0
        for i, layer_idx in enumerate(trainable_indices):
            layer_name = f"Layer_{layer_idx}"
            # Distribute trainable layers vertically with equal spacing
            if num_trainable > 1:
                y_pos = vertical_spacing * (i - (num_trainable - 1) / 2)
            else:
                y_pos = vertical_spacing / 2  # Single trainable layer
            positions[layer_name] = (layer_idx + 1, y_pos)
    
    # Position non-trainable layers (below the center line)
    if num_non_trainable > 0:
        vertical_spacing = 0.7  # Less spacing for non-trainable
        for i, layer_idx in enumerate(non_trainable_indices):
            layer_name = f"Layer_{layer_idx}"
            # Distribute non-trainable layers vertically with equal spacing
            if num_non_trainable > 1:
                y_pos = -vertical_spacing * (i - (num_non_trainable - 1) / 2)
            else:
                y_pos = -vertical_spacing / 2  # Single non-trainable layer
            positions[layer_name] = (layer_idx + 1, y_pos)
    
    # Get edge weights for drawing
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw the network with weighted edges
    nodes = nx.draw_networkx_nodes(
        G,
        positions,
        ax=ax_net,
        node_color=[node_colors[node] for node in G.nodes()],
        node_size=[node_sizes[node] for node in G.nodes()],
    )
    
    # Draw edges with varying width based on weight
    edges = nx.draw_networkx_edges(
        G,
        positions,
        ax=ax_net,
        width=[w * 1.5 for w in edge_weights],  # Scale width
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        connectionstyle='arc3,rad=0.1',  # Curved edges
        arrowstyle='-|>',
        alpha=0.7
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G,
        positions,
        ax=ax_net,
        labels=layer_labels,
        font_size=10,
        font_weight='bold',
        font_color='black'
    )
    
    # Remove axes
    ax_net.set_axis_off()
    
    # Add a legend
    trainable_patch = mpatches.Patch(color='lightgreen', label='Trainable')
    non_trainable_patch = mpatches.Patch(color='lightsalmon', label='Non-trainable')
    io_patch = mpatches.Patch(color='lightblue', label='I/O')
    ax_net.legend(handles=[trainable_patch, non_trainable_patch, io_patch], loc='upper right')
    
    # Display parameter details based on network size
    if show_detailed:
        # Remove the existing ax_params for detailed visualization
        fig.delaxes(ax_params)
        
        # Get trainable layers
        trainable_layers = [layer for layer in neural_network.layers if layer.trainable]
        if trainable_layers:
            # Create a more flexible gridspec for parameters with proper spacing
            n_layers = len(trainable_layers)
            
            # Adjust figure size based on number of layers for better display
            if n_layers > 1:
                fig.set_size_inches(figsize[0], figsize[1] * 1.5)
            
            # For small networks, use a single column with more vertical space
            n_rows, n_cols = n_layers, 1
            
            # Create gridspec with large spacing between subplots
            param_gs = GridSpec(n_rows, n_cols, top=0.65, bottom=0.05, figure=fig, 
                                hspace=1.0, wspace=0.4)
            
            for i, layer in enumerate(trainable_layers):
                layer_idx = neural_network.layers.index(layer)
                layer_type = layer.__class__.__name__
                
                # Calculate the grid position
                if n_cols == 1:
                    ax = fig.add_subplot(param_gs[i, 0])
                else:
                    row_idx = i // n_cols
                    col_idx = i % n_cols
                    ax = fig.add_subplot(param_gs[row_idx, col_idx])
                    
                ax.set_title(f"Layer {layer_idx}: {layer_type} Parameters", fontsize=12)
                
                # Display parameters based on layer type
                if hasattr(layer, 'plot_params'):
                    layer.plot_params(ax)
                elif layer_type == "FullyConnected":
                    # Plot weights as a heatmap
                    if hasattr(layer, 'weights'):
                        # Only show weights excluding bias row
                        weights_display = layer.weights[:-1, :]
                        im = ax.imshow(weights_display, cmap='viridis')
                        ax.set_xlabel('Output Neurons')
                        ax.set_ylabel('Input Connections')
                        
                        # Add colorbar with proper sizing
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.1)
                        plt.colorbar(im, cax=cax, label='Weight Value')
                        
                        # Plot bias as part of the same heatmap instead of separate axis
                        # Add a row label for bias
                        ax.set_yticks(list(range(weights_display.shape[0])) + [weights_display.shape[0]])
                        ax.set_yticklabels([f"In {i}" for i in range(weights_display.shape[0])] + ["Bias"])
                        
                        # Add the bias values as text below the heatmap
                        bias = layer.weights[-1, :]
                        for j, b in enumerate(bias):
                            ax.text(j, weights_display.shape[0] + 0.3, f"{b:.2f}", 
                                   ha="center", va="center", color="white",
                                   bbox=dict(boxstyle="round,pad=0.3", fc="royalblue", alpha=0.7))
                elif layer_type == "Conv":
                    # For Conv layers, show a grid of kernels
                    if hasattr(layer, 'weights'):
                        kernels = layer.weights
                        num_kernels = kernels.shape[0]
                        
                        # Instead of creating many subplots, show kernels as a grid in one plot
                        # Display only a sample of kernels if there are many
                        max_display = min(9, num_kernels)
                        fig_size = int(np.ceil(np.sqrt(max_display)))
                        
                        # Create a grid to display kernels within the current ax
                        divider = make_axes_locatable(ax)
                        
                        kernel_grid = []
                        for k in range(max_display):
                            if k == 0:
                                kernel_ax = ax
                            else:
                                kernel_ax = divider.append_axes("right" if k % 2 == 0 else "bottom", 
                                                             size="100%", pad=0.7)
                            kernel_grid.append(kernel_ax)
                            
                        # Create a single composite image of kernels
                        kernel_height, kernel_width = 0, 0
                        
                        # Determine kernel dimensions
                        if len(kernels.shape) == 4:  # 2D convolution
                            kernel_height, kernel_width = kernels.shape[1:3]
                        else:  # 1D convolution
                            kernel_height = kernels.shape[1]
                            kernel_width = 1
                            
                        # Create a composite image
                        composite = np.zeros((fig_size * kernel_height, fig_size * kernel_width))
                        
                        for k in range(max_display):
                            row = k // fig_size
                            col = k % fig_size
                            
                            if len(kernels.shape) == 4:  # 2D convolution
                                # Show first channel of each kernel
                                kernel_img = kernels[k, 0]
                                h, w = kernel_img.shape
                                composite[row*h:(row+1)*h, col*w:(col+1)*w] = kernel_img
                            else:  # 1D convolution
                                kernel_img = kernels[k, 0]
                                h = len(kernel_img)
                                composite[row*h:(row+1)*h, col:col+1] = kernel_img.reshape(-1, 1)
                        
                        # Display the composite image
                        img = ax.imshow(composite, cmap='viridis')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.1)
                        plt.colorbar(img, cax=cax)
                        
                        ax.set_title(f"Kernels (showing {max_display}/{num_kernels})")
                        ax.axis('off')
                        
                        # Display bias values directly on the plot rather than as a separate axis
                        if hasattr(layer, 'bias'):
                            ax_text = fig.add_subplot(param_gs[i, 0]) if n_cols == 1 else fig.add_subplot(param_gs[row_idx, col_idx+1])
                            ax_text.set_title(f"Biases for {num_kernels} kernels")
                            ax_text.axis('tight')
                            ax_text.axis('off')
                            
                            # Create a table with bias values
                            cell_text = [[f"{b:.3f}" for b in layer.bias]]
                            row_labels = ['Bias']
                            col_labels = [f"K{i}" for i in range(len(layer.bias))]
                            
                            # Color-code the bias values
                            cell_colors = [['#d4f1f9' if b >= 0 else '#ffcccc' for b in layer.bias]]
                            
                            ax_text.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                                      cellColours=cell_colors, loc='center', cellLoc='center')
    else:
        # Simple table of parameters for large networks
        ax_params.axis('tight')
        ax_params.axis('off')
        
        table_data = []
        table_columns = ['Layer', 'Type', 'Shape', 'Parameters']
        
        for i, layer in enumerate(neural_network.layers):
            if layer.trainable:
                layer_type = layer.__class__.__name__
                
                if hasattr(layer, 'get_params_count') and hasattr(layer, 'get_params_shapes'):
                    params_count = layer.get_params_count()
                    shapes_info = layer.get_params_shapes()
                elif layer_type == "Conv":
                    weight_shape = layer.weights.shape
                    num_params = np.prod(weight_shape) + len(layer.bias)
                    shapes_info = f"W: {weight_shape}, b: {layer.bias.shape}"
                    params_count = num_params
                elif layer_type == "FullyConnected":
                    weight_shape = layer.weights.shape
                    num_params = np.prod(weight_shape)
                    shapes_info = f"W: {weight_shape}"
                    params_count = num_params
                else:
                    continue
                
                table_data.append([f"Layer_{i}", layer_type, shapes_info, f"{params_count:,}"])
        
        if table_data:
            ax_params.table(
                cellText=table_data,
                colLabels=table_columns,
                loc='center',
                cellLoc='center',
                colColours=['#f0f0f0'] * len(table_columns)
            )
    
    # Set titles
    fig.suptitle(title, fontsize=18)
    
    # Add total parameters count
    fig.text(0.5, 0.01, f"Total Trainable Parameters: {total_params:,}", ha='center', fontsize=14)
    
    # Instead of tight_layout, manually adjust the figure
    # This avoids the warning message and gives more control over spacing
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.6, wspace=0.4)
    
    # Display the figure if requested
    if display:
        plt.figure(fig.number)
        plt.draw()
        plt.pause(0.001)  # Short pause to allow the figure to be displayed
    
    return fig
