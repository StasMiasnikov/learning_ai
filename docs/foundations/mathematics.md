# Mathematics for AI

Mathematics forms the backbone of artificial intelligence and machine learning. This section covers the essential mathematical concepts you need to understand and work effectively with LLMs and multi-agent systems.

## 📊 Linear Algebra

Linear algebra is fundamental to understanding how neural networks and language models process and transform information.

### Vectors and Vector Operations

**Vector Basics**:
```python
import numpy as np

# Creating vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition and subtraction
v_sum = v1 + v2      # [5, 7, 9]
v_diff = v1 - v2     # [-3, -3, -3]

# Scalar multiplication
v_scaled = 2 * v1    # [2, 4, 6]
```

**Dot Product (Inner Product)**:
```python
# Dot product: measures similarity between vectors
dot_product = np.dot(v1, v2)  # 32
# Alternative: v1 @ v2 or np.sum(v1 * v2)

# Geometric interpretation: v1 · v2 = |v1| |v2| cos(θ)
magnitude_v1 = np.linalg.norm(v1)  # sqrt(14)
magnitude_v2 = np.linalg.norm(v2)  # sqrt(77)
```

**Applications in AI**:
- **Attention Mechanisms**: Dot products compute attention weights
- **Similarity Measures**: Cosine similarity uses normalized dot products
- **Neural Network Layers**: Linear transformations use vector operations

### Matrices and Matrix Operations

**Matrix Fundamentals**:
```python
# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Matrix dimensions
print(f"A shape: {A.shape}")  # (2, 3)
print(f"B shape: {B.shape}")  # (3, 2)

# Matrix multiplication
C = np.dot(A, B)  # or A @ B
print(f"C shape: {C.shape}")  # (2, 2)
```

**Special Matrices**:
```python
# Identity matrix
I = np.eye(3)  # 3x3 identity matrix

# Zero matrix
Z = np.zeros((2, 3))

# Transpose
A_T = A.T  # or A.transpose()

# Inverse (for square matrices)
square_matrix = np.array([[1, 2], [3, 4]])
inverse = np.linalg.inv(square_matrix)
```

**Matrix Operations in Neural Networks**:
```python
# Forward pass in neural network layer
def linear_layer(X, W, b):
    """
    X: input matrix (batch_size, input_dim)
    W: weight matrix (input_dim, output_dim)
    b: bias vector (output_dim,)
    """
    return X @ W + b

# Example
batch_size, input_dim, output_dim = 32, 784, 128
X = np.random.randn(batch_size, input_dim)
W = np.random.randn(input_dim, output_dim)
b = np.random.randn(output_dim)

output = linear_layer(X, W, b)
print(f"Output shape: {output.shape}")  # (32, 128)
```

### Eigenvalues and Eigenvectors

**Definition**:
For a square matrix A, if Av = λv for some non-zero vector v, then:
- v is an eigenvector of A
- λ is the corresponding eigenvalue

```python
# Computing eigenvalues and eigenvectors
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

**Applications**:
- **Principal Component Analysis (PCA)**: Uses eigenvectors of covariance matrix
- **Spectral Clustering**: Based on eigenvalues of graph Laplacian
- **Stability Analysis**: Eigenvalues determine system stability

### Matrix Decompositions

**Singular Value Decomposition (SVD)**:
```python
# SVD: A = UΣV^T
A = np.random.randn(4, 6)
U, s, Vt = np.linalg.svd(A)

print(f"U shape: {U.shape}")    # (4, 4)
print(f"s shape: {s.shape}")    # (4,)
print(f"Vt shape: {Vt.shape}")  # (6, 6)

# Reconstruct original matrix
A_reconstructed = U @ np.diag(s) @ Vt[:4, :]
```

**Applications**:
- **Dimensionality Reduction**: Low-rank approximations
- **Recommendation Systems**: Matrix factorization
- **Image Compression**: Truncated SVD

## 🧮 Calculus

Calculus is essential for understanding optimization algorithms and neural network training.

### Derivatives and Partial Derivatives

**Single Variable Derivatives**:
```python
import sympy as sp

# Define variable and function
x = sp.Symbol('x')
f = x**3 + 2*x**2 + x + 1

# Compute derivative
f_prime = sp.diff(f, x)
print(f"f'(x) = {f_prime}")  # 3*x**2 + 4*x + 1

# Evaluate at specific point
value_at_2 = f_prime.subs(x, 2)
print(f"f'(2) = {value_at_2}")  # 21
```

**Partial Derivatives**:
```python
# Multivariable function
x, y = sp.symbols('x y')
f = x**2 + 3*x*y + y**2

# Partial derivatives
df_dx = sp.diff(f, x)  # 2*x + 3*y
df_dy = sp.diff(f, y)  # 3*x + 2*y

# Gradient vector
gradient = [df_dx, df_dy]
print(f"∇f = {gradient}")
```

### Chain Rule

**Mathematical Foundation**:
If y = f(u) and u = g(x), then dy/dx = (dy/du) × (du/dx)

```python
# Chain rule example
u = sp.Symbol('u')
x = sp.Symbol('x')

# u = x^2, y = sin(u)
u_func = x**2
y_func = sp.sin(u)

# Manual chain rule
dy_du = sp.diff(y_func, u)  # cos(u)
du_dx = sp.diff(u_func, x)  # 2*x

# dy/dx = dy/du × du/dx
dy_dx_manual = dy_du.subs(u, u_func) * du_dx
print(f"dy/dx (manual) = {dy_dx_manual}")

# Direct differentiation
y_full = sp.sin(x**2)
dy_dx_direct = sp.diff(y_full, x)
print(f"dy/dx (direct) = {dy_dx_direct}")
```

**Applications in Neural Networks**:
```python
# Backpropagation example: computing gradients
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Forward pass
x = 2.0
z1 = 3 * x      # z1 = 6
a1 = sigmoid(z1) # a1 = sigmoid(6)
z2 = 2 * a1     # z2 = 2 * sigmoid(6)
loss = z2       # Simple loss function

# Backward pass (chain rule)
dloss_dz2 = 1                              # ∂loss/∂z2
dz2_da1 = 2                               # ∂z2/∂a1
da1_dz1 = sigmoid_derivative(z1)          # ∂a1/∂z1
dz1_dx = 3                                # ∂z1/∂x

# Chain rule: ∂loss/∂x = ∂loss/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂x
dloss_dx = dloss_dz2 * dz2_da1 * da1_dz1 * dz1_dx

print(f"Gradient ∂loss/∂x = {dloss_dx}")
```

### Optimization

**Gradient Descent**:
```python
def gradient_descent(f, grad_f, x_start, learning_rate=0.01, iterations=1000):
    """
    Minimize function f using gradient descent
    
    Args:
        f: function to minimize
        grad_f: gradient of f
        x_start: starting point
        learning_rate: step size
        iterations: number of iterations
    """
    x = x_start
    history = [x]
    
    for i in range(iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x)
        
        if i % 100 == 0:
            print(f"Iteration {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
    
    return x, history

# Example: minimize f(x) = x^2 - 4x + 5
def f(x):
    return x**2 - 4*x + 5

def grad_f(x):
    return 2*x - 4

# Find minimum
x_min, history = gradient_descent(f, grad_f, x_start=0.0)
print(f"Minimum at x = {x_min:.4f}, f(x) = {f(x_min):.4f}")
```

**Multivariable Optimization**:
```python
# Gradient descent for multivariable functions
def gradient_descent_2d(grad_f, x_start, learning_rate=0.01, iterations=1000):
    x, y = x_start
    
    for i in range(iterations):
        grad_x, grad_y = grad_f(x, y)
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        
        if i % 100 == 0:
            print(f"Iteration {i}: ({x:.4f}, {y:.4f})")
    
    return x, y

# Example: minimize f(x,y) = x^2 + 2y^2 - 2xy + x - y
def grad_f_2d(x, y):
    grad_x = 2*x - 2*y + 1
    grad_y = 4*y - 2*x - 1
    return grad_x, grad_y

x_min, y_min = gradient_descent_2d(grad_f_2d, (0.0, 0.0))
print(f"Minimum at ({x_min:.4f}, {y_min:.4f})")
```

## 📈 Statistics and Probability

Understanding probability and statistics is crucial for working with uncertainty in AI systems.

### Probability Basics

**Fundamental Concepts**:
```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Probability mass function (discrete)
# Example: Fair coin flip
coin_outcomes = [0, 1]  # 0 = tails, 1 = heads
coin_probabilities = [0.5, 0.5]

# Binomial distribution: number of heads in n flips
n_flips = 10
p_heads = 0.5
binomial_dist = stats.binom(n_flips, p_heads)

# Probability of exactly 6 heads
prob_6_heads = binomial_dist.pmf(6)
print(f"P(6 heads in 10 flips) = {prob_6_heads:.4f}")

# Cumulative probability: P(X ≤ 6)
prob_at_most_6 = binomial_dist.cdf(6)
print(f"P(X ≤ 6) = {prob_at_most_6:.4f}")
```

**Normal Distribution**:
```python
# Standard normal distribution (μ=0, σ=1)
standard_normal = stats.norm(0, 1)

# Generate random samples
samples = standard_normal.rvs(1000)

# Probability density at specific points
x_values = np.linspace(-3, 3, 100)
pdf_values = standard_normal.pdf(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, 'b-', label='PDF')
plt.hist(samples, bins=30, density=True, alpha=0.7, label='Samples')
plt.legend()
plt.title('Standard Normal Distribution')
plt.show()
```

### Bayes' Theorem

**Mathematical Form**:
P(A|B) = P(B|A) × P(A) / P(B)

**Practical Example**:
```python
# Medical diagnosis example
# Disease D, Test T
# P(D) = 0.001 (disease prevalence)
# P(T+|D) = 0.99 (test sensitivity)
# P(T-|¬D) = 0.95 (test specificity)

def bayes_theorem(prior_disease, sensitivity, specificity, test_positive=True):
    """
    Calculate posterior probability using Bayes' theorem
    """
    if test_positive:
        # P(T+|D) × P(D)
        numerator = sensitivity * prior_disease
        # P(T+) = P(T+|D)×P(D) + P(T+|¬D)×P(¬D)
        evidence = sensitivity * prior_disease + (1 - specificity) * (1 - prior_disease)
    else:
        # P(T-|D) × P(D)
        numerator = (1 - sensitivity) * prior_disease
        # P(T-) = P(T-|D)×P(D) + P(T-|¬D)×P(¬D)
        evidence = (1 - sensitivity) * prior_disease + specificity * (1 - prior_disease)
    
    posterior = numerator / evidence
    return posterior

# Calculate probability of disease given positive test
prob_disease_given_positive = bayes_theorem(0.001, 0.99, 0.95, test_positive=True)
print(f"P(Disease|Positive Test) = {prob_disease_given_positive:.4f}")

# Calculate probability of disease given negative test
prob_disease_given_negative = bayes_theorem(0.001, 0.99, 0.95, test_positive=False)
print(f"P(Disease|Negative Test) = {prob_disease_given_negative:.6f}")
```

### Information Theory

**Entropy**:
```python
def entropy(probabilities):
    """Calculate Shannon entropy"""
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Example: fair coin
fair_coin = [0.5, 0.5]
fair_entropy = entropy(fair_coin)
print(f"Entropy of fair coin: {fair_entropy:.4f} bits")

# Example: biased coin
biased_coin = [0.9, 0.1]
biased_entropy = entropy(biased_coin)
print(f"Entropy of biased coin: {biased_entropy:.4f} bits")

# Example: uniform distribution over 8 outcomes
uniform_8 = [1/8] * 8
uniform_entropy = entropy(uniform_8)
print(f"Entropy of uniform distribution (8 outcomes): {uniform_entropy:.4f} bits")
```

**Cross-Entropy** (used in classification loss):
```python
def cross_entropy(true_probs, pred_probs):
    """Calculate cross-entropy loss"""
    return -np.sum([p * np.log2(q) for p, q in zip(true_probs, pred_probs) if q > 0])

# Example: true distribution vs predicted distribution
true_dist = [1, 0, 0]  # One-hot: class 0
pred_dist = [0.8, 0.15, 0.05]  # Model prediction

ce_loss = cross_entropy(true_dist, pred_dist)
print(f"Cross-entropy loss: {ce_loss:.4f}")

# Perfect prediction
perfect_pred = [1, 0, 0]
perfect_ce = cross_entropy(true_dist, perfect_pred)
print(f"Perfect prediction cross-entropy: {perfect_ce:.4f}")
```

**Kullback-Leibler Divergence**:
```python
def kl_divergence(p, q):
    """Calculate KL divergence D(P||Q)"""
    return np.sum([p_i * np.log2(p_i / q_i) for p_i, q_i in zip(p, q) if p_i > 0 and q_i > 0])

# Example: comparing two probability distributions
p = [0.5, 0.3, 0.2]
q = [0.4, 0.4, 0.2]

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

print(f"KL(P||Q) = {kl_pq:.4f}")
print(f"KL(Q||P) = {kl_qp:.4f}")
print("Note: KL divergence is not symmetric")
```

## 🔢 Discrete Mathematics

### Graph Theory

**Basic Graph Operations**:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes and edges
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])

# Graph properties
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Degree of node 1: {G.degree[1]}")

# Shortest path
shortest_path = nx.shortest_path(G, 1, 4)
print(f"Shortest path from 1 to 4: {shortest_path}")

# Visualize
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=16, font_weight='bold')
plt.title('Simple Graph')
plt.show()
```

**Applications in AI**:
- **Knowledge Graphs**: Representing relationships between entities
- **Neural Network Architectures**: Graphs represent connections
- **Social Network Analysis**: Understanding relationships and influence
- **Recommendation Systems**: Bipartite graphs of users and items

### Combinatorics

**Permutations and Combinations**:
```python
import math
from itertools import permutations, combinations

# Permutations: order matters
# P(n, r) = n! / (n-r)!
def permutation(n, r):
    return math.factorial(n) // math.factorial(n - r)

# Combinations: order doesn't matter  
# C(n, r) = n! / (r! × (n-r)!)
def combination(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Examples
print(f"P(5, 3) = {permutation(5, 3)}")  # 60
print(f"C(5, 3) = {combination(5, 3)}")  # 10

# Generate actual permutations and combinations
items = ['A', 'B', 'C', 'D']
perms = list(permutations(items, 3))
combs = list(combinations(items, 3))

print(f"Permutations of 3 from {items}: {perms[:5]}...")
print(f"Combinations of 3 from {items}: {combs}")
```

## 🧪 Practical Applications in AI

### Neural Network Mathematics

**Forward Propagation**:
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def relu(x):
    return np.maximum(0, x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0)
        
        return dW1, db1, dW2, db2

# Example usage
nn = SimpleNeuralNetwork(4, 8, 1)
X = np.random.randn(100, 4)
y = np.random.randint(0, 2, (100, 1))

output = nn.forward(X)
gradients = nn.backward(X, y, output)
print("Neural network forward and backward pass completed")
```

### Attention Mechanism Mathematics

**Scaled Dot-Product Attention**:
```python
def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot-product attention
    
    Args:
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Calculate attention scores
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Apply attention weights to values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 2, 5, 64
Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

## 📚 Study Resources

### Textbooks
1. **"Linear Algebra and Its Applications"** by David C. Lay
2. **"Calculus: Early Transcendentals"** by James Stewart
3. **"All of Statistics"** by Larry Wasserman
4. **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman

### Online Courses
1. **3Blue1Brown** - Excellent visual explanations of linear algebra and calculus
2. **Khan Academy** - Comprehensive coverage of all mathematical topics
3. **MIT OpenCourseWare** - Linear Algebra (18.06) and Multivariable Calculus (18.02)
4. **StatQuest** - Clear explanations of statistical concepts

### Programming Resources
1. **NumPy Documentation** - Essential for numerical computing
2. **SciPy Documentation** - Scientific computing library
3. **SymPy Tutorial** - Symbolic mathematics in Python
4. **Matplotlib Tutorials** - Data visualization

## 🎯 Practice Problems

### Linear Algebra Problems

**Problem 1**: Matrix Operations
```python
# Given matrices A and B, compute the following:
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 4],
              [2, 5],
              [3, 6]])

# 1. A @ B
# 2. B @ A  
# 3. A.T @ A
# 4. Eigenvalues of A.T @ A
```

**Problem 2**: SVD Application
```python
# Use SVD to compress an image
from PIL import Image

# Load image and convert to grayscale
img = np.array(Image.open('your_image.jpg').convert('L'))

# Apply SVD
U, s, Vt = np.linalg.svd(img)

# Reconstruct with different numbers of components
for k in [10, 50, 100]:
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    compression_ratio = k * (U.shape[0] + Vt.shape[1]) / (img.shape[0] * img.shape[1])
    print(f"Components: {k}, Compression ratio: {compression_ratio:.2f}")
```

### Calculus Problems

**Problem 3**: Optimization
```python
# Find the minimum of f(x, y) = x^2 + y^2 - 2x - 4y + 5
# using gradient descent

def f(x, y):
    return x**2 + y**2 - 2*x - 4*y + 5

def gradient_f(x, y):
    df_dx = 2*x - 2
    df_dy = 2*y - 4
    return df_dx, df_dy

# Implement gradient descent and find the minimum
```

### Probability Problems

**Problem 4**: Bayes' Theorem
```python
# A spam filter has the following properties:
# - 95% of spam emails contain the word "offer"
# - 5% of legitimate emails contain the word "offer"  
# - 40% of all emails are spam
# 
# If an email contains "offer", what's the probability it's spam?
```

## ✅ Knowledge Check

Before proceeding, ensure you can:

1. **Perform matrix operations** and understand their geometric interpretation
2. **Calculate derivatives** and apply the chain rule
3. **Implement gradient descent** for simple optimization problems
4. **Apply Bayes' theorem** to real-world scenarios
5. **Calculate entropy** and information-theoretic measures
6. **Use NumPy and SciPy** for mathematical computations
7. **Understand the mathematical foundations** of neural networks

## 🚀 Next Steps

With solid mathematical foundations, you're ready to explore:

1. **[Programming Essentials](programming.md)** - Master the programming tools
2. **[Deep Learning Basics](deep-learning.md)** - Apply mathematics to neural networks
3. **[LLM Architecture](../llms/architecture.md)** - Understand transformer mathematics

---

*Mathematics is the language of artificial intelligence. These concepts will be applied throughout your journey with LLMs and multi-agent systems.*
