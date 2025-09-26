# Deep Learning Basics for LLM Development

Deep learning forms the foundation of modern Language Models and Multi-Agent systems. This section provides hands-on understanding of neural networks, focusing on concepts essential for working with LLMs and agent architectures.

## 🧠 Neural Network Fundamentals

### Perceptron: The Building Block

**Mathematical Foundation**:
```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Simple perceptron implementation"""
    
    def __init__(self, input_size: int, learning_rate: float = 0.01):
        # Initialize weights randomly
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
    
    def activation(self, x: float) -> int:
        """Step activation function"""
        return 1 if x >= 0 else 0
    
    def forward(self, inputs: np.ndarray) -> int:
        """Forward pass"""
        # Calculate weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        return self.activation(weighted_sum)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the perceptron"""
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                prediction = self.forward(X[i])
                error = y[i] - prediction
                total_error += abs(error)
                
                # Update weights (Perceptron learning rule)
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
            
            if total_error == 0:
                print(f"Converged after {epoch + 1} epochs")
                break

# Example: Learn AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(X_and, y_and)

# Test the trained perceptron
for i, x in enumerate(X_and):
    prediction = perceptron.forward(x)
    print(f"Input: {x}, Expected: {y_and[i]}, Predicted: {prediction}")
```

### Multi-Layer Perceptrons (MLPs)

**Deep Network Implementation**:
```python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return (x > 0).astype(float)

class MLP:
    """Multi-layer perceptron with backpropagation"""
    
    def __init__(self, layer_sizes: list, activation: str = 'sigmoid'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation functions
        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation_func = relu
            self.activation_derivative = relu_derivative
    
    def forward(self, X):
        """Forward propagation"""
        self.layer_outputs = [X]
        
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            
            # Apply activation function (except for output layer)
            if i < self.num_layers - 2:  # Hidden layers
                a = self.activation_func(z)
            else:  # Output layer
                a = sigmoid(z)  # Always use sigmoid for output
            
            self.layer_outputs.append(a)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """Backpropagation"""
        m = X.shape[0]  # Number of samples
        
        # Calculate output layer error
        output_error = self.layer_outputs[-1] - y
        
        # Initialize gradient lists
        weight_gradients = []
        bias_gradients = []
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Calculate gradients for current layer
            if i == self.num_layers - 2:  # Output layer
                delta = output_error * sigmoid_derivative(self.layer_outputs[i + 1])
            else:  # Hidden layers
                delta = np.dot(delta, self.weights[i + 1].T) * \
                       self.activation_derivative(self.layer_outputs[i + 1])
            
            # Calculate weight and bias gradients
            weight_grad = np.dot(self.layer_outputs[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calculate loss (mean squared error)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((predictions > 0.5) == y) * 100
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

# Example: XOR problem (non-linearly separable)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Create and train MLP
mlp = MLP([2, 4, 1], activation='relu')  # 2 inputs, 4 hidden units, 1 output
losses = mlp.train(X_xor, y_xor, epochs=1000, learning_rate=0.1)

# Test predictions
predictions = mlp.predict(X_xor)
print("\nXOR Results:")
for i, x in enumerate(X_xor):
    pred = predictions[i][0]
    expected = y_xor[i][0]
    print(f"Input: {x}, Expected: {expected}, Predicted: {pred:.3f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True)
plt.show()
```

## 🔄 Convolutional Neural Networks (CNNs)

While primarily used for images, understanding CNNs helps with multi-modal LLM applications.

### Basic CNN Operations

**Convolution and Pooling**:
```python
def convolution_2d(image, kernel, stride=1, padding=0):
    """Perform 2D convolution operation"""
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    # Calculate output dimensions
    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    
    # Perform convolution
    output = np.zeros((output_height, output_width))
    
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            # Extract region
            region = image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            # Apply kernel
            output[i // stride, j // stride] = np.sum(region * kernel)
    
    return output

def max_pooling_2d(image, pool_size=2, stride=2):
    """Perform 2D max pooling"""
    output_height = image.shape[0] // stride
    output_width = image.shape[1] // stride
    
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            # Extract pooling region
            start_i, end_i = i * stride, (i + 1) * stride
            start_j, end_j = j * stride, (j + 1) * stride
            region = image[start_i:end_i, start_j:end_j]
            
            # Take maximum
            output[i, j] = np.max(region)
    
    return output

# Example usage
image = np.random.rand(8, 8)
edge_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

# Apply convolution
conv_output = convolution_2d(image, edge_kernel, padding=1)
pooled_output = max_pooling_2d(conv_output)

print(f"Original image shape: {image.shape}")
print(f"After convolution: {conv_output.shape}")
print(f"After max pooling: {pooled_output.shape}")
```

**Simple CNN Implementation**:
```python
class SimpleCNN:
    """Basic CNN for understanding convolution layers"""
    
    def __init__(self):
        # Define some basic kernels
        self.edge_kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]) / 8
        
        self.blur_kernel = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]]) / 9
    
    def apply_kernel(self, image, kernel):
        """Apply kernel to image"""
        return convolution_2d(image, kernel, padding=1)
    
    def feature_extraction(self, image):
        """Extract features using multiple kernels"""
        features = []
        
        # Apply different kernels
        edge_features = self.apply_kernel(image, self.edge_kernel)
        blur_features = self.apply_kernel(image, self.blur_kernel)
        
        features.extend([edge_features, blur_features])
        return features
    
    def forward(self, image):
        """Simple forward pass"""
        # Extract features
        features = self.feature_extraction(image)
        
        # Apply pooling to each feature map
        pooled_features = []
        for feature_map in features:
            pooled = max_pooling_2d(feature_map)
            pooled_features.append(pooled)
        
        return pooled_features

# Example
cnn = SimpleCNN()
sample_image = np.random.rand(32, 32)
features = cnn.forward(sample_image)

print(f"Number of feature maps: {len(features)}")
for i, feature_map in enumerate(features):
    print(f"Feature map {i} shape: {feature_map.shape}")
```

## 🔄 Recurrent Neural Networks (RNNs)

RNNs are crucial for understanding sequence processing, which forms the basis of language models.

### Vanilla RNN Implementation

**Basic RNN Cell**:
```python
class RNNCell:
    """Single RNN cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        # Input to hidden
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        # Hidden to hidden  
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Hidden to output
        self.Why = np.random.randn(hidden_size, input_size) * 0.01
        # Biases
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, input_size))
    
    def forward(self, x, h_prev):
        """Forward pass through RNN cell"""
        # h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + bh)
        h = np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        
        # y_t = h_t @ Why + by
        y = h @ self.Why + self.by
        
        return h, y
    
    def backward(self, dh_next, dy, h, h_prev, x):
        """Backward pass through RNN cell"""
        # Output gradients
        dWhy = h.T @ dy
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # Hidden gradients  
        dh = dy @ self.Why.T + dh_next
        dh_raw = (1 - h * h) * dh  # tanh derivative
        
        # Input gradients
        dWxh = x.T @ dh_raw
        dWhh = h_prev.T @ dh_raw
        dbh = np.sum(dh_raw, axis=0, keepdims=True)
        dx = dh_raw @ self.Wxh.T
        dh_prev = dh_raw @ self.Whh.T
        
        return dx, dh_prev, dWxh, dWhh, dWhy, dbh, dby

class SimpleRNN:
    """Simple RNN for sequence processing"""
    
    def __init__(self, input_size: int, hidden_size: int, sequence_length: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Create RNN cell
        self.cell = RNNCell(input_size, hidden_size)
    
    def forward(self, X):
        """Forward pass through sequence"""
        batch_size = X.shape[0]
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        
        # Store states for backprop
        self.hidden_states = [h]
        self.outputs = []
        
        # Process sequence
        for t in range(self.sequence_length):
            x_t = X[:, t, :]  # Input at time t
            h, y = self.cell.forward(x_t, h)
            
            self.hidden_states.append(h)
            self.outputs.append(y)
        
        return np.array(self.outputs).transpose(1, 0, 2)  # (batch, time, features)
    
    def train_step(self, X, y, learning_rate=0.001):
        """Single training step"""
        # Forward pass
        predictions = self.forward(X)
        
        # Calculate loss
        loss = np.mean((predictions - y) ** 2)
        
        # Backward pass
        dh_next = np.zeros((X.shape[0], self.hidden_size))
        
        # Initialize gradients
        dWxh = np.zeros_like(self.cell.Wxh)
        dWhh = np.zeros_like(self.cell.Whh)
        dWhy = np.zeros_like(self.cell.Why)
        dbh = np.zeros_like(self.cell.bh)
        dby = np.zeros_like(self.cell.by)
        
        # Backpropagate through time
        for t in reversed(range(self.sequence_length)):
            dy = predictions[:, t, :] - y[:, t, :]
            
            dx, dh_next, dWxh_t, dWhh_t, dWhy_t, dbh_t, dby_t = self.cell.backward(
                dh_next, dy, self.hidden_states[t + 1], self.hidden_states[t], X[:, t, :]
            )
            
            # Accumulate gradients
            dWxh += dWxh_t
            dWhh += dWhh_t
            dWhy += dWhy_t
            dbh += dbh_t
            dby += dby_t
        
        # Update weights
        self.cell.Wxh -= learning_rate * dWxh
        self.cell.Whh -= learning_rate * dWhh
        self.cell.Why -= learning_rate * dWhy
        self.cell.bh -= learning_rate * dbh
        self.cell.by -= learning_rate * dby
        
        return loss

# Example: Simple sequence prediction
def generate_sine_sequence(seq_length, num_samples):
    """Generate sine wave sequences"""
    X = []
    y = []
    
    for _ in range(num_samples):
        start = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(start, start + 2 * np.pi, seq_length + 1)
        sequence = np.sin(t)
        
        X.append(sequence[:-1].reshape(-1, 1))  # Input sequence
        y.append(sequence[1:].reshape(-1, 1))   # Target sequence (shifted by 1)
    
    return np.array(X), np.array(y)

# Train RNN on sine prediction
X_train, y_train = generate_sine_sequence(seq_length=20, num_samples=100)
rnn = SimpleRNN(input_size=1, hidden_size=10, sequence_length=20)

print("Training RNN on sine wave prediction...")
for epoch in range(100):
    loss = rnn.train_step(X_train, y_train, learning_rate=0.01)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Long Short-Term Memory (LSTM)

**LSTM Cell Implementation**:
```python
class LSTMCell:
    """LSTM cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for gates
        # Forget gate
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # Candidate values
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h_prev, c_prev):
        """Forward pass through LSTM cell"""
        # Concatenate input and previous hidden state
        combined = np.hstack((x, h_prev))
        
        # Forget gate: decides what information to discard
        f_t = self.sigmoid(combined @ self.Wf + self.bf)
        
        # Input gate: decides what new information to store
        i_t = self.sigmoid(combined @ self.Wi + self.bi)
        
        # Candidate values: new information that could be stored
        c_tilde = np.tanh(combined @ self.Wc + self.bc)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate: decides what parts of cell state to output
        o_t = self.sigmoid(combined @ self.Wo + self.bo)
        
        # Hidden state
        h_t = o_t * np.tanh(c_t)
        
        # Store intermediate values for backprop
        self.cache = {
            'f_t': f_t, 'i_t': i_t, 'c_tilde': c_tilde, 'o_t': o_t,
            'c_t': c_t, 'h_t': h_t, 'c_prev': c_prev, 'h_prev': h_prev,
            'combined': combined
        }
        
        return h_t, c_t

class SimpleLSTM:
    """Simple LSTM for sequence processing"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.cell = LSTMCell(input_size, hidden_size)
        
        # Output projection layer
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
        
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, X):
        """Forward pass through LSTM sequence"""
        batch_size, seq_length, input_size = X.shape
        
        # Initialize states
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        self.states = []  # Store for backprop
        
        for t in range(seq_length):
            h, c = self.cell.forward(X[:, t, :], h, c)
            
            # Project to output space
            y = h @ self.Wy + self.by
            outputs.append(y)
            
            # Store states
            self.states.append((h.copy(), c.copy()))
        
        return np.array(outputs).transpose(1, 0, 2)

# Example: Character-level language modeling
def create_char_dataset(text, seq_length):
    """Create character-level sequences"""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create sequences
    X, y = [], []
    for i in range(len(text) - seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + 1:i + seq_length + 1]
        
        X.append([char_to_idx[ch] for ch in seq_in])
        y.append([char_to_idx[ch] for ch in seq_out])
    
    return np.array(X), np.array(y), char_to_idx, idx_to_char

# Simple text for demonstration
sample_text = "hello world this is a simple text for lstm training"
X, y, char_to_idx, idx_to_char = create_char_dataset(sample_text, seq_length=10)

# Convert to one-hot encoding
vocab_size = len(char_to_idx)
X_onehot = np.zeros((X.shape[0], X.shape[1], vocab_size))
y_onehot = np.zeros((y.shape[0], y.shape[1], vocab_size))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X_onehot[i, j, X[i, j]] = 1
        y_onehot[i, j, y[i, j]] = 1

print(f"Dataset shape: {X_onehot.shape}")
print(f"Vocabulary size: {vocab_size}")
```

## ⚡ Attention Mechanisms

Attention is the core innovation behind Transformers and modern LLMs.

### Basic Attention Implementation

**Scaled Dot-Product Attention**:
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention mechanism
    
    Args:
        Q: Query matrix (batch_size, seq_len_q, d_k)
        K: Key matrix (batch_size, seq_len_k, d_k)  
        V: Value matrix (batch_size, seq_len_v, d_v)
        mask: Optional mask (batch_size, seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output (batch_size, seq_len_q, d_v)
        attention_weights: Attention weights (batch_size, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # Calculate attention scores
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Apply attention weights to values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x):
    """Stable softmax implementation"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """Combine the head dimension back"""
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """Forward pass through multi-head attention"""
        batch_size = query.shape[0]
        
        # Linear projections
        Q = query @ self.W_q
        K = key @ self.W_k  
        V = value @ self.W_v
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply attention to each head
        attention_output = np.zeros_like(Q)
        attention_weights = np.zeros((batch_size, self.num_heads, Q.shape[2], K.shape[2]))
        
        for head in range(self.num_heads):
            head_output, head_weights = scaled_dot_product_attention(
                Q[:, head, :, :], K[:, head, :, :], V[:, head, :, :], mask
            )
            attention_output[:, head, :, :] = head_output
            attention_weights[:, head, :, :] = head_weights
        
        # Combine heads
        combined_output = self.combine_heads(attention_output)
        
        # Final linear projection
        output = combined_output @ self.W_o
        
        return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 2, 10, 64
num_heads = 8

# Create sample input
x = np.random.randn(batch_size, seq_len, d_model)

# Initialize multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass
output, weights = mha.forward(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

# Visualize attention weights for first head
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(weights[0, 0], cmap='Blues')
plt.colorbar()
plt.title('Attention Weights (First Head)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()
```

## 🏗️ Transformer Architecture

The Transformer is the foundation of modern LLMs.

### Transformer Building Blocks

**Position Encoding**:
```python
def positional_encoding(seq_len: int, d_model: int):
    """Create positional encoding matrix"""
    pos_encoding = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    
    return pos_encoding

class LayerNormalization:
    """Layer normalization implementation"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)  # Learnable scale
        self.beta = np.zeros(d_model)  # Learnable shift
    
    def forward(self, x):
        """Forward pass through layer norm"""
        # Calculate mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output

class FeedForward:
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """Forward pass through feed-forward network"""
        # First linear layer with ReLU activation
        hidden = relu(x @ self.W1 + self.b1)
        
        # Second linear layer
        output = hidden @ self.W2 + self.b2
        
        return output

class TransformerBlock:
    """Single Transformer block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout_rate = 0.1
    
    def forward(self, x, mask=None):
        """Forward pass through transformer block"""
        # Multi-head self-attention with residual connection
        attn_output, attn_weights = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)  # Add & Norm
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)  # Add & Norm
        
        return x, attn_weights

class SimpleTransformer:
    """Simple Transformer implementation"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_seq_len: int):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, d_ff))
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0  # True for allowed positions, False for masked
    
    def forward(self, input_ids):
        """Forward pass through transformer"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = np.zeros((batch_size, seq_len, self.d_model))
        for i in range(batch_size):
            for j in range(seq_len):
                x[i, j] = self.token_embedding[input_ids[i, j]]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :]
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward(x, mask=causal_mask)
            all_attention_weights.append(attn_weights)
        
        # Output projection to vocabulary
        logits = x @ self.output_projection
        
        return logits, all_attention_weights

# Example: Create and test simple transformer
vocab_size = 100
d_model = 64
num_heads = 8
num_layers = 2
d_ff = 256
max_seq_len = 20

transformer = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

# Test with random input
batch_size, seq_len = 2, 10
input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))

logits, attention_weights = transformer.forward(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Number of attention weight matrices: {len(attention_weights)}")
print(f"Each attention weight shape: {attention_weights[0].shape}")
```

## 🎯 Training Deep Networks

### Optimization Algorithms

**Advanced Optimizers**:
```python
class SGDOptimizer:
    """Stochastic Gradient Descent with momentum"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params, grads, param_name):
        """Update parameters using SGD with momentum"""
        if param_name not in self.velocities:
            self.velocities[param_name] = np.zeros_like(params)
        
        # Update velocity
        self.velocities[param_name] = (self.momentum * self.velocities[param_name] - 
                                     self.lr * grads)
        
        # Update parameters
        return params + self.velocities[param_name]

class AdamOptimizer:
    """Adam optimizer implementation"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params, grads, param_name):
        """Update parameters using Adam optimizer"""
        self.t += 1
        
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(params)
            self.v[param_name] = np.zeros_like(params)
        
        # Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grads
        
        # Update biased second raw moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Training loop example
def train_model(model, X_train, y_train, optimizer, epochs=100, batch_size=32):
    """Generic training loop"""
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            # Calculate loss
            loss = np.mean((predictions - y_batch) ** 2)
            epoch_loss += loss
            
            # Backward pass (simplified - would need actual gradients)
            # This is where you'd calculate actual gradients
            grads = 2 * (predictions - y_batch) / batch_size
            
            # Update parameters (example for one parameter)
            # In practice, you'd update all parameters
            if hasattr(model, 'weights'):
                model.weights = optimizer.update(model.weights, grads, 'weights')
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return losses
```

## 📚 Practical Exercises

### Exercise 1: Build a Character-Level RNN

**Task**: Implement a character-level RNN that can generate text after training on a corpus.

```python
class CharRNN:
    """Character-level RNN for text generation"""
    
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TODO: Initialize RNN layers, output projection
        # Your implementation here
    
    def forward(self, input_seq, hidden_state=None):
        """Forward pass through character RNN"""
        # TODO: Implement forward pass
        pass
    
    def generate(self, seed_char, length=100):
        """Generate text starting from seed character"""
        # TODO: Implement text generation
        pass
    
    def train(self, text_data, epochs=100):
        """Train the character RNN"""
        # TODO: Implement training loop
        pass

# Test your implementation
text = "your favorite book text here"
char_rnn = CharRNN(vocab_size=len(set(text)), hidden_size=128)
char_rnn.train(text)
generated_text = char_rnn.generate('T', length=200)
print(generated_text)
```

### Exercise 2: Implement Attention Visualization

**Task**: Create a tool to visualize attention weights in a simple attention mechanism.

```python
class AttentionVisualizer:
    """Visualize attention patterns"""
    
    def __init__(self):
        self.attention_weights = None
    
    def compute_attention(self, query, key, value):
        """Compute attention and store weights"""
        # TODO: Implement attention computation
        pass
    
    def visualize_attention(self, input_tokens, output_tokens):
        """Create attention heatmap"""
        # TODO: Create visualization using matplotlib
        pass
    
    def analyze_attention_patterns(self):
        """Analyze common attention patterns"""
        # TODO: Implement pattern analysis
        pass

# Test attention visualization
sentences = ["The cat sat on the mat", "Hello world example"]
visualizer = AttentionVisualizer()
# Test your implementation
```

### Exercise 3: Mini Transformer Implementation

**Task**: Build a minimal transformer that can perform a simple task (e.g., copy sequence, reverse sequence).

```python
class MiniTransformer:
    """Minimal transformer for educational purposes"""
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2):
        # TODO: Initialize transformer components
        pass
    
    def forward(self, input_ids):
        """Forward pass through mini transformer"""
        # TODO: Implement forward pass
        pass
    
    def train_copy_task(self, max_seq_len=10, num_samples=1000):
        """Train transformer to copy input sequences"""
        # TODO: Generate copy task data and train
        pass
    
    def evaluate(self, test_sequences):
        """Evaluate transformer performance"""
        # TODO: Implement evaluation
        pass

# Test your mini transformer
transformer = MiniTransformer(vocab_size=20)
transformer.train_copy_task()
```

## 📊 Evaluation and Metrics

### Model Evaluation Techniques

**Common Metrics for Deep Learning**:
```python
def calculate_metrics(y_true, y_pred, task_type='classification'):
    """Calculate various metrics for model evaluation"""
    
    if task_type == 'classification':
        # Classification metrics
        accuracy = np.mean(y_pred == y_true)
        
        # Confusion matrix
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))
        
        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
        
        # Precision, Recall, F1 (simplified for binary classification)
        if len(unique_labels) == 2:
            tp = confusion_matrix[1, 1]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix
            }
        else:
            return {
                'accuracy': accuracy,
                'confusion_matrix': confusion_matrix
            }
    
    elif task_type == 'regression':
        # Regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

# Perplexity for language models
def calculate_perplexity(probabilities):
    """Calculate perplexity for language model evaluation"""
    log_probs = np.log(np.clip(probabilities, 1e-10, 1.0))
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity

# BLEU score for text generation (simplified)
def calculate_bleu_score(reference, candidate, n=4):
    """Simplified BLEU score calculation"""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    scores = []
    for i in range(1, n+1):
        ref_ngrams = get_ngrams(ref_tokens, i)
        cand_ngrams = get_ngrams(cand_tokens, i)
        
        if len(cand_ngrams) == 0:
            scores.append(0)
            continue
        
        # Count matches
        matches = 0
        for ngram in cand_ngrams:
            if ngram in ref_ngrams:
                matches += 1
        
        precision = matches / len(cand_ngrams)
        scores.append(precision)
    
    # Geometric mean
    if all(score > 0 for score in scores):
        bleu = np.exp(np.mean(np.log(scores)))
    else:
        bleu = 0
    
    # Brevity penalty
    bp = min(1, np.exp(1 - len(ref_tokens) / len(cand_tokens))) if len(cand_tokens) > 0 else 0
    
    return bleu * bp

# Example usage
reference = "the cat sat on the mat"
candidate = "the cat is on the mat"
bleu = calculate_bleu_score(reference, candidate)
print(f"BLEU Score: {bleu:.3f}")
```

## ✅ Knowledge Check

Before proceeding to LLM-specific content, ensure you understand:

1. **Neural Network Fundamentals**: Forward/backward propagation, activation functions, loss functions
2. **RNN Concepts**: Sequence processing, hidden states, vanishing gradients, LSTM/GRU
3. **Attention Mechanisms**: Scaled dot-product attention, multi-head attention, self-attention
4. **Transformer Architecture**: Position encoding, layer normalization, residual connections
5. **Training Techniques**: Optimization algorithms, regularization, evaluation metrics
6. **Implementation Skills**: Can implement basic networks from scratch in NumPy

## 🚀 Next Steps

With deep learning foundations established, you're ready for:

1. **[LLM Architecture](../llms/architecture.md)** - Dive deep into transformer-based language models
2. **[Training Process](../llms/training.md)** - Understand large-scale LLM training
3. **[Building LLM Agents](../agents/architecture.md)** - Apply deep learning to agent systems

---

*Deep learning concepts are the technical foundation for understanding and implementing modern LLM agents and multi-agent systems. Master these fundamentals before proceeding to specialized LLM architectures.*
