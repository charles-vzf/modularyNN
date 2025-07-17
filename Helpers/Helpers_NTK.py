import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import seaborn as sns
import time

def validate_network_for_ntk(network):
    """
    Validate that the network is suitable for NTK computation.
    
    Args:
        network: The neural network to validate
        
    Returns:
        bool: True if valid, raises ValueError if not
    """
    if not hasattr(network, 'layers') or len(network.layers) == 0:
        raise ValueError("Network must have layers")
    
    # Check for at least one trainable layer
    trainable_layers = [layer for layer in network.layers if hasattr(layer, 'trainable') and layer.trainable]
    if len(trainable_layers) == 0:
        raise ValueError("Network must have at least one trainable layer for NTK computation")
    
    # Check that trainable layers have weights
    for layer in trainable_layers:
        if not hasattr(layer, 'weights'):
            raise ValueError(f"Trainable layer {layer.__class__.__name__} must have weights attribute")
    
    return True

def compute_jacobian(network, x, image_size=None):
    """
    Compute the Jacobian of the network output with respect to the parameters.
    
    Args:
        network: The neural network
        x: Input sample (single sample)
        image_size: Size of the input images (optional, set to None for non-image data like Iris)
        
    Returns:
        Jacobian matrix of shape (output_dim, total_params)
    """
    # Validate network before computation
    validate_network_for_ntk(network)
    # Check the layer structure to determine correct input handling
    has_flatten_layer = any(layer.__class__.__name__ == 'Flatten' for layer in network.layers)
    
    # Prepare a single input sample - handle Iris dataset case
    if hasattr(x, 'shape') and len(x.shape) == 1:
        # For 1D inputs like Iris features (no need for image reshaping)
        if has_flatten_layer:
            # Pass a proper batch for networks with Flatten layers
            # The Flatten layer expects a tensor with at least 2 dimensions
            # For Iris data, we need to keep it as a batch of features
            x_input = x.reshape(1, -1)  # Just make it a batch of 1
        else:
            x_input = x.reshape(1, -1)  # Just make it a batch of 1
    elif hasattr(x, 'shape') and len(x.shape) >= 2:
        # For 2D+ inputs that might be images
        if image_size is not None:
            x_input = x.reshape(1, 1, image_size, image_size)  # MNIST-like reshape
        else:
            # If no image_size, assume it's already in the right shape with batch dimension
            x_input = x.reshape(1, -1) if x.shape[0] != 1 else x
    else:
        # Handle non-array inputs
        print(f"Warning: Unexpected input type: {type(x)}")
        x_input = np.array([x]).reshape(1, -1)
    
    # Get the output dimension by running a test forward pass
    # Store original network state
    original_training = getattr(network, 'training', None)
    if hasattr(network, 'training'):
        network.training = False  # Set to evaluation mode
    
    # Run forward pass to determine output shape
    test_input = x_input.copy()
    for layer in network.layers:
        test_input = layer.forward(test_input)
    output_dim = test_input.shape[-1]
    
    # Restore original network state
    if original_training is not None:
        network.training = original_training

    
    # Get the total number of parameters
    total_params = 0
    for layer in network.layers:
        if hasattr(layer, 'weights') and layer.trainable:
            total_params += np.prod(layer.weights.shape)
            # Note: For FullyConnected layers, bias is included in weights matrix (last row)
            # No need to add separate bias parameters
    
    # Compute Jacobian for each output dimension
    jacobian = np.zeros((output_dim, total_params))
    
    # For each output dimension
    for i in range(output_dim):
        # Create a one-hot vector for this output dimension
        output_grad = np.zeros((1, output_dim))
        output_grad[0, i] = 1.0
        
        # Reset parameter gradients
        for layer in network.layers:
            if hasattr(layer, 'grad_weights'):
                layer.grad_weights = None
            if hasattr(layer, 'grad_bias'):
                layer.grad_bias = None
        
        # Forward pass through all layers
        input_tensor = x_input
        for layer_idx, layer in enumerate(network.layers):
            try:
                input_tensor = layer.forward(input_tensor)
            except ValueError as e:
                print(f"Error in layer {layer.__class__.__name__}")
                print(f"Input tensor shape: {input_tensor.shape}")
                if hasattr(layer, 'weights'):
                    print(f"Weights shape: {layer.weights.shape}")
                raise e
        
        # Backward pass with the one-hot gradient
        error_tensor = output_grad
        for layer in reversed(network.layers):
            error_tensor = layer.backward(error_tensor)
        
        # Collect the gradients with improved numerical stability
        param_index = 0
        for layer in network.layers:
            if hasattr(layer, 'weights') and layer.trainable:
                # Get gradients with improved handling
                if layer.grad_weights is not None:
                    # Check for numerical issues before clipping
                    if np.any(np.isnan(layer.grad_weights)) or np.any(np.isinf(layer.grad_weights)):
                        print(f"Warning: NaN or Inf gradients detected in layer {layer.__class__.__name__}")
                        flat_grad = np.zeros(np.prod(layer.weights.shape))
                    else:
                        # Apply conservative gradient clipping
                        grad_weights_clipped = np.clip(layer.grad_weights, -1e6, 1e6)
                        flat_grad = grad_weights_clipped.flatten()
                else:
                    flat_grad = np.zeros(np.prod(layer.weights.shape))
                
                jacobian[i, param_index:param_index + len(flat_grad)] = flat_grad
                param_index += len(flat_grad)
                
                # Note: For FullyConnected layers, bias is included in weights matrix
                # No separate bias handling needed
    
    return jacobian


def compute_ntk(network, x1, image_size=None, x2=None, verbose=True, batch_size=None, normalize=True):
    """
    Compute the Neural Tangent Kernel between two sets of inputs.
    
    Args:
        network: The neural network
        image_size: Size of the input images (optional, set to None for non-image data like Iris)
        x1: First set of inputs
        x2: Second set of inputs (optional, if None, use x1)
        verbose: Whether to print progress (default: True)
        batch_size: Batch size for computing NTK (optional, for memory efficiency)
        normalize: Whether to normalize the kernel matrix (default: True)
        
    Returns:
        NTK matrix of shape (len(x1), len(x2))
    """
    # Input validation
    if network is None:
        raise ValueError("Network cannot be None")
    if x1 is None or len(x1) == 0:
        raise ValueError("x1 cannot be None or empty")
    if not hasattr(network, 'layers') or len(network.layers) == 0:
        raise ValueError("Network must have layers")
    
    # Validate network architecture
    validate_network_for_ntk(network)
    # If x2 is not provided, use x1
    if x2 is None:
        x2 = x1
        is_symmetric = True
    else:
        # Check if x1 and x2 are the same (content-wise, not just object identity)
        is_symmetric = np.array_equal(x1, x2) if hasattr(x1, 'shape') and hasattr(x2, 'shape') else x1 is x2
    
    n1 = len(x1)
    n2 = len(x2)
    
    if verbose:
        print(f"Computing NTK matrix of shape ({n1}, {n2})")
        start_time = time.time()
    
    # Initialize the NTK matrix
    ntk = np.zeros((n1, n2))
    
    # Store original network state if available
    original_training = getattr(network, 'training', None)
    if hasattr(network, 'training'):
        network.training = False  # Set to evaluation mode
    
    # Initialize progress counters
    processed = 0
    total_pairs = n1 * n2 if not is_symmetric else n1 * (n1 + 1) // 2
    
    # Batched computation for memory efficiency
    if batch_size is not None and batch_size > 0:
        # Pre-compute all Jacobians with memory management
        if verbose:
            print("Pre-computing Jacobians...")
        
        jacobians1 = []
        for i in range(n1):
            if verbose and i % max(1, n1 // 10) == 0:
                print(f"Computing Jacobian {i+1}/{n1}")
            try:
                jac = compute_jacobian(network, x1[i], image_size)
                if jac is not None and not np.any(np.isnan(jac)):
                    jacobians1.append(jac)
                else:
                    print(f"Warning: Invalid Jacobian at index {i}, using zero matrix")
                    # Create zero matrix with correct shape based on previous valid Jacobians
                    if len(jacobians1) > 0:
                        jacobians1.append(np.zeros_like(jacobians1[0]))
                    else:
                        # Fallback: compute a test Jacobian to get the shape
                        test_jac = compute_jacobian(network, x1[0], image_size)
                        jacobians1.append(np.zeros_like(test_jac) if test_jac is not None else np.zeros((1, 1)))
            except Exception as e:
                print(f"Error computing Jacobian at index {i}: {e}")
                # Use zero matrix as fallback
                if len(jacobians1) > 0:
                    jacobians1.append(np.zeros_like(jacobians1[0]))
                else:
                    jacobians1.append(np.zeros((1, 1)))  # Minimal fallback
        
        # If x2 is different from x1, compute its Jacobians as well
        if not is_symmetric:
            jacobians2 = []
            for j in range(n2):
                if verbose and j % max(1, n2 // 10) == 0:
                    print(f"Computing Jacobian {j+1}/{n2}")
                try:
                    jac = compute_jacobian(network, x2[j], image_size)
                    if jac is not None and not np.any(np.isnan(jac)):
                        jacobians2.append(jac)
                    else:
                        print(f"Warning: Invalid Jacobian at index {j}, using zero matrix")
                        if len(jacobians2) > 0:
                            jacobians2.append(np.zeros_like(jacobians2[0]))
                        else:
                            jacobians2.append(np.zeros_like(jacobians1[0]) if len(jacobians1) > 0 else np.zeros((1, 1)))
                except Exception as e:
                    print(f"Error computing Jacobian at index {j}: {e}")
                    if len(jacobians2) > 0:
                        jacobians2.append(np.zeros_like(jacobians2[0]))
                    else:
                        jacobians2.append(np.zeros_like(jacobians1[0]) if len(jacobians1) > 0 else np.zeros((1, 1)))
        else:
            jacobians2 = jacobians1
        
        # Compute NTK with batching
        for i in range(n1):
            j_start = i if is_symmetric else 0
            
            for j_batch_start in range(j_start, n2, batch_size):
                j_batch_end = min(j_batch_start + batch_size, n2)
                
                # Process batch
                for j in range(j_batch_start, j_batch_end):
                    # Compute the NTK value using inner product of flattened Jacobians
                    jac1_flat = jacobians1[i].flatten()
                    jac2_flat = jacobians2[j].flatten()
                    
                    # Check shape compatibility
                    if jac1_flat.shape != jac2_flat.shape:
                        print(f"Warning: Jacobian shape mismatch at ({i}, {j}): {jac1_flat.shape} vs {jac2_flat.shape}")
                        ntk_value = 0.0
                    else:
                        # Compute dot product with numerical stability check
                        try:
                            ntk_value = np.dot(jac1_flat, jac2_flat)
                            
                            # Check for numerical issues
                            if np.isnan(ntk_value) or np.isinf(ntk_value):
                                print(f"Warning: NaN or Inf detected at ({i}, {j})")
                                ntk_value = 0.0
                        except Exception as e:
                            print(f"Error computing NTK at ({i}, {j}): {e}")
                            ntk_value = 0.0
                    
                    ntk[i, j] = ntk_value
                    
                    # Use symmetry for efficiency when appropriate
                    if is_symmetric and i != j:
                        ntk[j, i] = ntk_value
                    
                    # Update progress
                    processed += 1
                    if verbose and processed % max(1, total_pairs // 50) == 0:
                        elapsed = time.time() - start_time
                        eta = (elapsed / processed) * (total_pairs - processed) if processed > 0 else 0
                        print(f"Progress: {processed}/{total_pairs} pairs ({100.0*processed/total_pairs:.1f}%), ETA: {eta:.1f}s")
    else:
        # Standard computation without batching
        for i in range(n1):
            if verbose and i % max(1, n1 // 10) == 0:
                print(f"Computing NTK row {i+1}/{n1}")
            
            # Compute Jacobian for the first input with error handling
            try:
                jacobian1 = compute_jacobian(network, x1[i], image_size)
                if jacobian1 is None or np.any(np.isnan(jacobian1)):
                    print(f"Warning: Invalid Jacobian for x1[{i}], skipping row")
                    continue
            except Exception as e:
                print(f"Error computing Jacobian for x1[{i}]: {e}, skipping row")
                continue
            
            j_start = i if is_symmetric else 0
            for j in range(j_start, n2):
                # Compute Jacobian for the second input with error handling
                try:
                    jacobian2 = compute_jacobian(network, x2[j], image_size)
                    if jacobian2 is None or np.any(np.isnan(jacobian2)):
                        print(f"Warning: Invalid Jacobian for x2[{j}], setting NTK to 0")
                        ntk[i, j] = 0.0
                        if is_symmetric and i != j:
                            ntk[j, i] = 0.0
                        continue
                except Exception as e:
                    print(f"Error computing Jacobian for x2[{j}]: {e}, setting NTK to 0")
                    ntk[i, j] = 0.0
                    if is_symmetric and i != j:
                        ntk[j, i] = 0.0
                    continue
                
                # Compute the NTK as the inner product of flattened Jacobians
                jac1_flat = jacobian1.flatten()
                jac2_flat = jacobian2.flatten()
                
                # Check shape compatibility
                if jac1_flat.shape != jac2_flat.shape:
                    print(f"Warning: Jacobian shape mismatch at ({i}, {j}): {jac1_flat.shape} vs {jac2_flat.shape}")
                    ntk_value = 0.0
                else:
                    try:
                        ntk_value = np.dot(jac1_flat, jac2_flat)
                        
                        # Check for numerical issues
                        if np.isnan(ntk_value) or np.isinf(ntk_value):
                            print(f"Warning: NaN or Inf detected at ({i}, {j})")
                            ntk_value = 0.0
                    except Exception as e:
                        print(f"Error computing NTK at ({i}, {j}): {e}")
                        ntk_value = 0.0
                
                ntk[i, j] = ntk_value
                
                # Use symmetry for efficiency when appropriate
                if is_symmetric and i != j:
                    ntk[j, i] = ntk_value
                
                # Update progress
                processed += 1
                if verbose and processed % max(1, total_pairs // 50) == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / processed) * (total_pairs - processed) if processed > 0 else 0
                    print(f"Progress: {processed}/{total_pairs} pairs ({100.0*processed/total_pairs:.1f}%), ETA: {eta:.1f}s")
    
    # Restore original network state
    if original_training is not None:
        network.training = original_training
    
    # Normalize the kernel matrix for numerical stability
    if normalize:
        max_abs_val = np.max(np.abs(ntk))
        if max_abs_val > 0 and np.isfinite(max_abs_val):
            ntk = ntk / max_abs_val
            if verbose:
                print(f"Normalized NTK by factor: {max_abs_val:.2e}")
        elif verbose:
            print("Warning: Could not normalize NTK matrix (max value is 0 or infinite)")
    
    if verbose:
        total_time = time.time() - start_time
        print(f"NTK computation completed in {total_time:.2f}s")
        # Quick check of the NTK properties
        print(f"NTK min: {np.min(ntk):.6f}, max: {np.max(ntk):.6f}, mean: {np.mean(ntk):.6f}")
        print(f"NTK is symmetric: {np.allclose(ntk, ntk.T, rtol=1e-5, atol=1e-8)}")
        
        # Check condition number with better error handling
        try:
            # Use scipy's more robust eigenvalue computation for large matrices
            if ntk.shape[0] > 100:
                # For large matrices, use a subset for condition number estimation
                sample_size = min(100, ntk.shape[0])
                idx = np.random.choice(ntk.shape[0], sample_size, replace=False)
                ntk_sample = ntk[np.ix_(idx, idx)]
                eigenvals = np.linalg.eigvals(ntk_sample)
            else:
                eigenvals = np.linalg.eigvals(ntk)
            
            # Filter out near-zero and invalid eigenvalues
            eigenvals = eigenvals[np.isfinite(eigenvals)]
            eigenvals = eigenvals[eigenvals > 1e-12]
            
            if len(eigenvals) > 1:
                cond_num = np.max(eigenvals) / np.min(eigenvals)
                print(f"Condition number: {cond_num:.2e}")
            else:
                print("Warning: Matrix appears to be singular or has insufficient valid eigenvalues")
        except Exception as e:
            print(f"Could not compute condition number: {e}")
    
    return ntk

def analyze_ntk_matrix(ntk_matrix, plot=True, save_plots=False, prefix="ntk_analysis"):
    """
    Analyzes and visualizes a Neural Tangent Kernel matrix.
    
    Args:
        ntk_matrix: The NTK matrix to analyze
        plot: Whether to generate and show plots (default: True)
        save_plots: Whether to save plots to files (default: False)
        prefix: Prefix for saved plot filenames (default: "ntk_analysis")
        
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {}
    
    # Basic properties
    results['shape'] = ntk_matrix.shape
    results['min_value'] = np.min(ntk_matrix)
    results['max_value'] = np.max(ntk_matrix)
    results['mean_value'] = np.mean(ntk_matrix)
    results['median_value'] = np.median(ntk_matrix)
    results['std_value'] = np.std(ntk_matrix)
    results['diagonal_mean'] = np.mean(np.diag(ntk_matrix))
    results['off_diagonal_mean'] = np.mean(ntk_matrix - np.diag(np.diag(ntk_matrix)))
    
    # Symmetry check
    results['is_symmetric'] = np.allclose(ntk_matrix, ntk_matrix.T, rtol=1e-5, atol=1e-8)
    if not results['is_symmetric']:
        max_asymmetry = np.max(np.abs(ntk_matrix - ntk_matrix.T))
        results['max_asymmetry'] = max_asymmetry
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(ntk_matrix)  # Use eigvalsh for symmetric matrices
    results['eigenvalues'] = eigenvalues
    results['min_eigenvalue'] = np.min(eigenvalues)
    results['max_eigenvalue'] = np.max(eigenvalues)
    results['is_positive_definite'] = np.all(eigenvalues > 0)
    results['is_positive_semidefinite'] = np.all(eigenvalues >= -1e-10)  # Numerical tolerance
    results['condition_number'] = np.max(eigenvalues) / np.max([np.min(np.abs(eigenvalues)), 1e-15])
    
    # Rank analysis
    rank_threshold = 1e-10
    effective_rank = np.sum(eigenvalues > rank_threshold)
    results['effective_rank'] = effective_rank
    results['rank_ratio'] = effective_rank / len(eigenvalues)
    
    # Sparsity analysis
    sparsity_threshold = 1e-6
    sparsity = np.sum(np.abs(ntk_matrix) < sparsity_threshold) / ntk_matrix.size
    results['sparsity'] = sparsity
    
    # Diagonal dominance
    diag_vals = np.diag(ntk_matrix)
    max_off_diag = np.max(np.abs(ntk_matrix - np.diag(diag_vals)))
    results['diagonal_dominance'] = np.min(np.abs(diag_vals)) / max_off_diag if max_off_diag > 0 else float('inf')
    
    # Print summary
    print("=== NTK Matrix Analysis ===")
    print(f"Shape: {results['shape']}")
    print(f"Value range: [{results['min_value']:.6f}, {results['max_value']:.6f}]")
    print(f"Mean value: {results['mean_value']:.6f}")
    print(f"Std deviation: {results['std_value']:.6f}")
    print(f"Diagonal mean: {results['diagonal_mean']:.6f}")
    print(f"Off-diagonal mean: {results['off_diagonal_mean']:.6f}")
    print(f"Symmetric: {results['is_symmetric']}")
    
    if not results['is_symmetric']:
        print(f"Maximum asymmetry: {results['max_asymmetry']:.6e}")
    
    print("\n=== Eigenvalue Analysis ===")
    print(f"Minimum eigenvalue: {results['min_eigenvalue']:.6e}")
    print(f"Maximum eigenvalue: {results['max_eigenvalue']:.6e}")
    print(f"Condition number: {results['condition_number']:.6e}")
    print(f"Positive definite: {results['is_positive_definite']}")
    print(f"Positive semidefinite: {results['is_positive_semidefinite']}")
    print(f"Effective rank: {results['effective_rank']} / {len(eigenvalues)} ({results['rank_ratio']:.2%})")
    
    print("\n=== Structure Analysis ===")
    print(f"Sparsity: {results['sparsity']:.2%}")
    print(f"Diagonal dominance: {results['diagonal_dominance']:.6f}")
    
    # Health check
    print("\n=== NTK Health Check ===")
    if results['condition_number'] > 1e10:
        print("WARNING: Matrix is ill-conditioned, may cause numerical issues")
    
    if not results['is_positive_semidefinite']:
        print("WARNING: Matrix is not positive semidefinite, may cause issues in kernel methods")
    
    if results['diagonal_dominance'] > 100:
        print("WARNING: Matrix is highly diagonal dominant, might not capture relationships well")
    
    if results['rank_ratio'] < 0.5:
        print("WARNING: Matrix has low effective rank, suggesting redundancy or poor feature extraction")
    
    if not plot:
        return results
    
    # Visualizations
    # 1. NTK Matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(ntk_matrix, cmap='viridis')
    plt.title('Neural Tangent Kernel Matrix')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{prefix}_heatmap.png", dpi=300)
    plt.show()
    
    # 2. Eigenvalue spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.sort(eigenvalues)[::-1], '-o', markersize=4)
    plt.title('Eigenvalue Spectrum of the NTK Matrix')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{prefix}_eigenspectrum.png", dpi=300)
    plt.show()
    
    # 3. Eigenvalue distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, alpha=0.7)
    plt.title('Distribution of Eigenvalues')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{prefix}_eigendist.png", dpi=300)
    plt.show()
    
    # 4. Off-diagonal distribution
    off_diag = ntk_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    plt.figure(figsize=(10, 6))
    plt.hist(off_diag.flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of Off-Diagonal Elements')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{prefix}_offdiag_dist.png", dpi=300)
    plt.show()
    
    # 5. Row norms (feature importance)
    row_norms = np.linalg.norm(ntk_matrix, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(row_norms, '-o', markersize=4)
    plt.title('Row Norms (Sample Influence)')
    plt.xlabel('Sample Index')
    plt.ylabel('L2 Norm')
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{prefix}_row_norms.png", dpi=300)
    plt.show()
    
    return results