# Pic trainable activation functions

Trainable activation functions represent a significant departure from traditional fixed activation functions in neural networks. While classical activations like ReLU, sigmoid, and tanh have fixed mathematical forms, trainable activations introduce learnable parameters that can adapt to the specific characteristics of the data and task at hand.


### Advantages

#### 1. Increased Expressivity
Trainable activations increase the expressivity of neural networks by allowing the activation landscape to adapt to the data distribution. This is particularly valuable when:
- Data has specific structural properties
- Different neurons need different activation characteristics  
- The optimal activation form is task-dependent

#### 2. Adaptive Receptive Fields
The $\theta$ parameter in Pic activation can be interpreted as controlling the "receptive field" of the neuron - where it is most sensitive to inputs. This relates to attention mechanisms in modern architectures.

#### 3. Gradient Flow Control
The smoothness parameter $\gamma$ allows fine control over gradient flow:
- $\gamma = 0$: Sharp transitions, potentially faster convergence but risk of gradient issues
- $\gamma = 1$: Smooth gradients, stable training but potentially slower convergence
- $\gamma \in (0,1)$: Balanced approach

### 4. Approximation Properties

The Pic activation function can approximate various common functions:

1. **Linear Function**: When $\theta \to 1$, the left part dominates
2. **Step Function**: When $\theta \to 0$ or $\theta \to 1$ with $\gamma = 0$
3. **Gaussian-like**: When $\gamma = 1$ with appropriate $\alpha$

Networks with trainable Pic activations maintain universal approximation properties. In fact, the adaptive nature may reduce the number of neurons required for certain approximation tasks.

**Theorem** (Informal): A feedforward network with trainable Pic activations can approximate any continuous function on $[0,1]^n$ with potentially fewer neurons than networks with fixed activations (for a same given precision)

## The Pic Activation Function

### Mathematical Definition

The Pic activation function $\text{pic}(x, \theta, \gamma)$ is defined as a parameterized piecewise function that maps the interval $[0,1]$ to $[0,1]$:

#### Piecewise Linear Version ($\gamma = 0$):

$$
\text{pic}(x, \theta) = \begin{cases}
\frac{x}{\theta} & \text{if } 0 \leq x \leq \theta \\
\frac{1-x}{1-\theta} & \text{if } \theta < x \leq 1
\end{cases}
$$

#### Smooth Version ($\gamma = 1$):

$$
\text{pic}_{\text{smooth}}(x, \theta) = \text{left}_{\text{part}} \times (1 - \sigma(\alpha(x - \theta))) + \text{right}_{\text{part}} \times \sigma(\alpha(x - \theta))
$$

where:
- $\text{left}_{\text{part}} = \frac{x}{\theta}$
- $\text{right}_{\text{part}} = \frac{1-x}{1-\theta}$
- $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the sigmoid function
- $\alpha$ controls the sharpness of the transition

#### Unified Version ($0 \leq \gamma \leq 1$):

$$
\text{pic}(x, \theta, \gamma) = (1-\gamma) \times \text{pic}_{\text{piecewise}}(x, \theta) + \gamma \times \text{pic}_{\text{smooth}}(x, \theta)
$$

### Key Properties

1. **Boundary Conditions**: $\text{pic}(0, \theta, \gamma) = 0$ and $\text{pic}(1, \theta, \gamma) = 0$
2. **Peak Property**: $\text{pic}(\theta, \theta, \gamma) = 1$
3. **Trainable Parameter**: $\theta \in (0,1)$ determines the peak location
4. **Smoothness Control**: $\gamma \in [0,1]$ controls differentiability
5. **Continuity**: The function is continuous everywhere
6. **Bounded Output**: Range is $[0,1]$ for input domain $[0,1]$

### Gradient Analysis

#### Gradient with respect to input $x$:

For the piecewise linear version ($\gamma = 0$):

$$
\frac{\partial \text{pic}}{\partial x} = \begin{cases}
\frac{1}{\theta} & \text{if } x \leq \theta \\
-\frac{1}{1-\theta} & \text{if } x > \theta
\end{cases}
$$

For the smooth version, the gradient is continuous and differentiable everywhere:

$$
\frac{\partial \text{pic}_{\text{smooth}}}{\partial x} = \frac{1}{\theta}(1 - \sigma) - \frac{1}{1-\theta}\sigma + (\text{right}_{\text{part}} - \text{left}_{\text{part}}) \times \alpha \sigma(1-\sigma)
$$

#### Gradient with respect to parameter $\theta$:

For the piecewise linear version:

$$
\frac{\partial \text{pic}}{\partial \theta} = \begin{cases}
-\frac{x}{\theta^2} & \text{if } x \leq \theta \\
\frac{1-x}{(1-\theta)^2} & \text{if } x > \theta
\end{cases}
$$

For the smooth version:

$$
\frac{\partial \text{pic}_{\text{smooth}}}{\partial \theta} = \frac{\partial \text{left}}{\partial \theta}(1-\sigma) + \frac{\partial \text{right}}{\partial \theta}\sigma + (\text{right} - \text{left}) \times (-\alpha \sigma(1-\sigma))
$$

where:
- $\frac{\partial \text{left}}{\partial \theta} = -\frac{x}{\theta^2}$
- $\frac{\partial \text{right}}{\partial \theta} = \frac{1-x}{(1-\theta)^2}$

## Related Work and Theoretical Foundation

### Learnable Activation Functions in Literature

#### 1. Parametric ReLU (PReLU) [He et al., 2015]:
$$\text{PReLU}(x) = \max(0, x) + \alpha \times \min(0, x)$$
where $\alpha$ is a learnable parameter.

#### 2. Exponential Linear Units (ELU) [Clevert et al., 2015]:
$$\text{ELU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(\exp(x) - 1) & \text{if } x \leq 0
\end{cases}$$

#### 3. Swish/SiLU [Ramachandran et al., 2017]:
$$\text{Swish}(x) = x \times \sigma(\beta x)$$
where $\beta$ can be learned or fixed.

#### 4. Adaptive Activation Functions [Jagtap et al., 2020]:
Various approaches to make activation functions adaptive to different layers and neurons.



### Optimization Landscape

The introduction of trainable parameters in activation functions affects the optimization landscape:

1. **Non-convexity**: The $\theta$ parameter introduces additional non-convexity
2. **Local Minima**: New local minima may emerge, but they often correspond to meaningful feature detectors
3. **Convergence**: Empirically, convergence properties remain good with proper initialization

## Computational Considerations

### Forward Pass Complexity
- Piecewise version: $O(1)$ per neuron
- Smooth version: $O(1)$ per neuron (with exponential computation)
- Unified version: $O(1)$ per neuron

### Backward Pass Complexity
- Additional gradient computation for $\theta$: $O(1)$ per neuron
- Memory overhead: One additional parameter per neuron/layer

### Numerical Stability

Key considerations:
1. **Clipping**: $\theta$ must be maintained in $(0,1)$ to avoid division by zero
2. **Smooth transitions**: High $\alpha$ values may cause numerical issues
3. **Gradient clipping**: May be beneficial for stable training

## Applications and Use Cases

### 1. Signal Processing
- Adaptive filtering with learnable center frequencies
- Peak detection in time series
- Spectral analysis with adaptive band-pass characteristics

### 2. Computer Vision
- Feature detectors with learnable spatial preferences
- Adaptive pooling based on image content
- Attention mechanisms in CNNs

### 3. Natural Language Processing
- Position-aware activations in transformers
- Adaptive attention based on sequence characteristics
- Language-specific activation patterns

### 4. Time Series Analysis
- Seasonal pattern detection with learnable periods
- Adaptive anomaly detection
- Financial modeling with regime-specific activations

## Implementation Considerations

### Various Strategies

1. **Random Initialization**: $\theta \sim \mathcal{U}(0.3, 0.7)$ to avoid extremes
2. **Data-driven**: Initialize based on data distribution statistics
3. **Layer-specific**: Different initialization for different layers
4. **Learning Rate Scheduling**: $\theta$ parameters may need different learning rates
5. **Progressive Training**: Start with fixed $\theta$, then make trainable
6. **Per-neuron $\theta$** vs **Per-layer $\theta$**
7. **Hierarchical $\theta$**: $\theta$ values organized in groups or structures
8. **Adaptative $\gamma$**: Allow $\gamma$ to be learned, potentially per layer or neuron
9. **Multi-peak $\theta$**: Allow multiple $\theta$ values per neuron for complex patterns

## References

1. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.

2. Clevert, D., et al. (2015). "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)." *ICLR*.

3. Ramachandran, P., et al. (2017). "Searching for Activation Functions." *arXiv preprint arXiv:1710.05941*.

4. Jagtap, A. D., et al. (2020). "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks." *Journal of Computational Physics*.

5. Apicella, A., et al. (2021). "A survey on modern trainable activation functions." *Neural Networks*.

6. Dubey, S. R., et al. (2022). "Activation functions in deep learning: A comprehensive survey and benchmark." *Neurocomputing*.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." *MIT Press*.

8. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553), 436-444.