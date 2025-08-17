# LLM Architecture: Understanding Transformer-Based Language Models

Large Language Models are built on the Transformer architecture, which revolutionized natural language processing. This section explores the detailed architecture of modern LLMs and how they process language.

## 🏗️ The Transformer Revolution

### From RNNs to Transformers

**Problems with RNNs for Language Modeling**:
- Sequential processing limits parallelization
- Vanishing gradients in long sequences
- Difficulty capturing long-range dependencies
- Limited scalability to very large datasets

**The Transformer Solution**:
- Parallel processing of all sequence positions
- Direct connections between any two positions
- Scalable to massive datasets and model sizes
- Foundation for GPT, BERT, T5, and modern LLMs

### Key Innovations

1. **Self-Attention**: Every token can attend to every other token
2. **Position Encoding**: Inject positional information without recurrence
3. **Layer Normalization**: Stabilize training of deep networks
4. **Residual Connections**: Enable training of very deep models

## 🔍 Deep Dive: Self-Attention Mechanism

### Mathematical Foundation

**Scaled Dot-Product Attention**:
```python
import numpy as np
import matplotlib.pyplot as plt

def scaled_dot_product_attention(Q, K, V, mask=None, temperature=1.0):
    """
    Implement scaled dot-product attention with detailed explanation
    
    Args:
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
        mask: Attention mask (optional)
        temperature: Scale factor for attention scores
    
    Returns:
        output: Attended values (batch_size, seq_len, d_v)
        attention_weights: Attention distribution (batch_size, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores
    # scores[i,j] = how much query i attends to key j
    scores = np.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(d_k) * temperature)
    print(f"Attention scores shape: {scores.shape}")
    
    # Step 2: Apply mask (for causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Step 3: Apply softmax to get attention weights
    # Convert scores to probabilities
    attention_weights = softmax(scores)
    print(f"Attention weights sum along last dim: {np.sum(attention_weights, axis=-1)[0, 0]:.3f}")
    
    # Step 4: Apply attention weights to values
    # Weighted sum of values based on attention
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example: Understanding attention patterns
def demonstrate_attention_patterns():
    """Show how attention works with concrete examples"""
    batch_size, seq_len, d_model = 1, 5, 4
    
    # Create simple embeddings for words: "The cat sat on mat"
    # Each word gets a unique embedding
    embeddings = np.array([[
        [1, 0, 0, 0],  # "The"
        [0, 1, 0, 0],  # "cat"  
        [0, 0, 1, 0],  # "sat"
        [0, 0, 0, 1],  # "on"
        [1, 1, 0, 0]   # "mat" (similar to "The" and "cat")
    ]])
    
    print("Word embeddings:")
    words = ["The", "cat", "sat", "on", "mat"]
    for i, word in enumerate(words):
        print(f"{word}: {embeddings[0, i]}")
    
    # Use embeddings as Q, K, V for self-attention
    output, attention_weights = scaled_dot_product_attention(
        embeddings, embeddings, embeddings
    )
    
    print(f"\nAttention weights (who attends to whom):")
    print("Rows=queries, Cols=keys")
    for i, query_word in enumerate(words):
        for j, key_word in enumerate(words):
            weight = attention_weights[0, i, j]
            print(f"{query_word}→{key_word}: {weight:.3f}", end=" ")
        print()
    
    return attention_weights[0]

# Run demonstration
attention_matrix = demonstrate_attention_patterns()
```

### Multi-Head Attention Implementation

**Complete Multi-Head Attention**:
```python
class MultiHeadAttention:
    """Production-ready multi-head attention implementation"""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Linear projections for Q, K, V (all heads computed together)
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, d_model))
        self.W_v = self._init_weights((d_model, d_model))
        
        # Output projection
        self.W_o = self._init_weights((d_model, d_model))
        
        # For storing attention weights during forward pass
        self.attention_weights = None
    
    def _init_weights(self, shape):
        """Xavier/Glorot initialization for transformer weights"""
        return np.random.randn(*shape) / np.sqrt(shape[0])
    
    def _reshape_for_heads(self, x):
        """
        Reshape tensor for multi-head attention
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        # First reshape to separate heads
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Then transpose to put heads dimension first
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x):
        """
        Combine multi-head outputs back to original shape
        (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        # Transpose back and reshape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention
        
        Args:
            query, key, value: Input tensors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        
        Returns:
            output: Attended values (batch_size, seq_len, d_model)
            attention_weights: Attention patterns (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # Step 1: Linear projections for all heads at once
        Q = query @ self.W_q  # (batch_size, seq_len, d_model)
        K = key @ self.W_k
        V = value @ self.W_v
        
        # Step 2: Reshape for multi-head attention
        Q = self._reshape_for_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self._reshape_for_heads(K)
        V = self._reshape_for_heads(V)
        
        # Step 3: Compute attention for all heads in parallel
        d_k = self.d_k
        
        # Attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for all heads if needed
            if mask.ndim == 2:  # (seq_len, seq_len)
                mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)
            elif mask.ndim == 3:  # (batch_size, seq_len, seq_len)
                mask = mask[:, np.newaxis, :, :]  # (batch_size, 1, seq_len, seq_len)
            
            scores = np.where(mask == 0, -np.inf, scores)
        
        # Softmax to get attention weights
        attention_weights = softmax(scores)
        self.attention_weights = attention_weights  # Store for analysis
        
        # Apply attention to values
        attended_values = np.matmul(attention_weights, V)
        
        # Step 4: Combine heads
        combined_output = self._combine_heads(attended_values)
        
        # Step 5: Final linear projection
        output = combined_output @ self.W_o
        
        return output, attention_weights
    
    def visualize_attention(self, tokens, head_idx=0, save_path=None):
        """Visualize attention patterns for a specific head"""
        if self.attention_weights is None:
            raise ValueError("No attention weights to visualize. Run forward() first.")
        
        weights = self.attention_weights[0, head_idx]  # First batch, specified head
        
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='Blues')
        plt.colorbar(label='Attention Weight')
        
        # Add token labels
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel('Keys (Attended To)')
        plt.ylabel('Queries (Attending From)')
        plt.title(f'Attention Patterns - Head {head_idx}')
        
        # Add weight values as text
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                plt.text(j, i, f'{weights[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if weights[i, j] > 0.5 else 'black')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage of multi-head attention
def test_multihead_attention():
    """Test multi-head attention with example sentence"""
    # Model parameters
    d_model, num_heads = 64, 8
    seq_len = 6
    batch_size = 1
    
    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create sample input (random embeddings for "Hello world this is test")
    tokens = ["Hello", "world", "this", "is", "a", "test"]
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = mha.forward(x, x, x)  # Self-attention
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention for first head
    mha.visualize_attention(tokens, head_idx=0)
    
    return mha, attention_weights

# Run test
mha_example, attn_weights = test_multihead_attention()
```

## 🏢 Complete Transformer Block

### Layer Components

**Layer Normalization**:
```python
class LayerNorm:
    """Layer normalization with learnable parameters"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)  # Shift parameter
        
        # For tracking statistics during training
        self.running_mean = np.zeros(d_model)
        self.running_var = np.ones(d_model)
    
    def forward(self, x, training=True):
        """
        Apply layer normalization
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            training: Whether in training mode
        """
        if training:
            # Compute statistics along the feature dimension
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
        else:
            # Use running statistics during inference
            mean = self.running_mean.reshape(1, 1, -1)
            var = self.running_var.reshape(1, 1, -1)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output
    
    def update_stats(self, x, momentum=0.9):
        """Update running statistics for inference"""
        batch_mean = np.mean(x, axis=(0, 1))
        batch_var = np.var(x, axis=(0, 1))
        
        self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
        self.running_var = momentum * self.running_var + (1 - momentum) * batch_var

class PositionwiseFeedForward:
    """Position-wise feed-forward network (FFN)"""
    
    def __init__(self, d_model: int, d_ff: int, activation='relu', dropout_rate: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Two linear transformations with activation in between
        self.W1 = self._init_weights((d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = self._init_weights((d_ff, d_model))
        self.b2 = np.zeros(d_model)
        
        # Activation function
        if activation == 'relu':
            self.activation = self._relu
        elif activation == 'gelu':
            self.activation = self._gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self, shape):
        """Initialize weights with appropriate scaling"""
        return np.random.randn(*shape) / np.sqrt(shape[0])
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _gelu(self, x):
        """Gaussian Error Linear Unit - used in GPT and BERT"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _dropout(self, x, training=True):
        """Apply dropout during training"""
        if not training or self.dropout_rate == 0:
            return x
        
        # Create dropout mask
        keep_prob = 1 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, x.shape)
        return x * mask / keep_prob
    
    def forward(self, x, training=True):
        """
        Forward pass through position-wise FFN
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            training: Whether in training mode
        """
        # First linear transformation + activation
        hidden = self.activation(x @ self.W1 + self.b1)
        
        # Apply dropout
        hidden = self._dropout(hidden, training)
        
        # Second linear transformation  
        output = hidden @ self.W2 + self.b2
        
        return output

class TransformerBlock:
    """Complete transformer block with attention and FFN"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Sub-layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, 'gelu', dropout_rate)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout_rate = dropout_rate
    
    def _dropout(self, x, training=True):
        """Apply dropout"""
        if not training or self.dropout_rate == 0:
            return x
        keep_prob = 1 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, x.shape)
        return x * mask / keep_prob
    
    def forward(self, x, mask=None, training=True):
        """
        Forward pass through transformer block
        
        Uses pre-norm architecture (norm before sub-layer) which is more stable
        """
        # Multi-head self-attention with residual connection
        # Pre-norm: normalize then apply attention
        norm_x = self.norm1.forward(x, training)
        attn_output, attention_weights = self.self_attention.forward(
            norm_x, norm_x, norm_x, mask
        )
        attn_output = self._dropout(attn_output, training)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection  
        # Pre-norm: normalize then apply FFN
        norm_x = self.norm2.forward(x, training)
        ff_output = self.feed_forward.forward(norm_x, training)
        ff_output = self._dropout(ff_output, training)
        x = x + ff_output  # Residual connection
        
        return x, attention_weights

# Test transformer block
def test_transformer_block():
    """Test complete transformer block"""
    d_model, num_heads, d_ff = 64, 8, 256
    seq_len, batch_size = 10, 2
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    
    # Create sample input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = transformer_block.forward(x, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify residual connections preserve shape
    assert output.shape == x.shape, "Transformer block changed tensor shape!"
    print("✓ Transformer block test passed!")

test_transformer_block()
```

## 🎯 Position Encoding

### Sinusoidal Position Encoding

**Understanding Position Encoding**:
```python
class PositionalEncoding:
    """Sinusoidal positional encoding for transformers"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        """Create the positional encoding matrix"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        
        # Create position indices
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        # Create dimension indices
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )
        
        # Apply sin to even dimensions
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd dimensions  
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_len = x.shape[1]
        
        # Add positional encoding to input
        # pe shape: (seq_len, d_model), broadcast to (batch_size, seq_len, d_model)
        x = x + self.pe[:seq_len, :]
        
        return x
    
    def visualize_encoding(self, max_pos=100):
        """Visualize positional encoding patterns"""
        pe_subset = self.pe[:max_pos, :]
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pe_subset.T, aspect='auto', cmap='RdBu')
        plt.colorbar(label='Encoding Value')
        plt.xlabel('Position')
        plt.ylabel('Embedding Dimension')
        plt.title('Positional Encoding Patterns')
        
        # Show that similar positions have similar encodings
        plt.figure(figsize=(12, 6))
        for pos in [10, 20, 30, 40]:
            plt.plot(self.pe[pos, :50], label=f'Position {pos}')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Encoding Value')
        plt.title('Positional Encodings for Different Positions')
        plt.legend()
        plt.show()
        
        # Show relative position relationships
        plt.figure(figsize=(12, 6))
        pos1, pos2 = 10, 15
        similarity = np.dot(self.pe[pos1], self.pe[pos2]) / (
            np.linalg.norm(self.pe[pos1]) * np.linalg.norm(self.pe[pos2])
        )
        print(f"Similarity between position {pos1} and {pos2}: {similarity:.3f}")
        
        # Show how similarity varies with distance
        base_pos = 50
        distances = []
        similarities = []
        
        for offset in range(-20, 21):
            if base_pos + offset >= 0 and base_pos + offset < self.max_seq_len:
                sim = np.dot(self.pe[base_pos], self.pe[base_pos + offset]) / (
                    np.linalg.norm(self.pe[base_pos]) * np.linalg.norm(self.pe[base_pos + offset])
                )
                distances.append(offset)
                similarities.append(sim)
        
        plt.plot(distances, similarities, 'o-')
        plt.xlabel('Relative Position Offset')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Position Similarity Relative to Position {base_pos}')
        plt.grid(True)
        plt.show()

# Test positional encoding
pos_encoding = PositionalEncoding(d_model=64, max_seq_len=100)
pos_encoding.visualize_encoding()
```

### Alternative Position Encoding Schemes

**Learned Position Embeddings**:
```python
class LearnedPositionalEmbedding:
    """Learned positional embeddings (like BERT)"""
    
    def __init__(self, d_model: int, max_seq_len: int):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Initialize learnable position embeddings
        self.position_embeddings = np.random.randn(max_seq_len, d_model) * 0.02
    
    def forward(self, x):
        """Add learned positional embeddings"""
        seq_len = x.shape[1]
        
        # Add position embeddings
        x = x + self.position_embeddings[:seq_len, :]
        
        return x
    
    def update_embeddings(self, gradients, learning_rate=0.001):
        """Update position embeddings during training"""
        self.position_embeddings -= learning_rate * gradients

class RotaryPositionalEncoding:
    """Rotary Position Embedding (RoPE) - used in modern LLMs"""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create frequency matrix
        self.freq_cis = self._create_frequency_matrix()
    
    def _create_frequency_matrix(self):
        """Create complex frequency matrix for rotary encoding"""
        # Create frequencies for each dimension pair
        freqs = 1.0 / (10000 ** (np.arange(0, self.d_model, 2) / self.d_model))
        
        # Create position indices
        t = np.arange(self.max_seq_len)
        
        # Create frequency matrix: (seq_len, d_model//2)
        freqs_for_each_token = np.outer(t, freqs)
        
        # Convert to complex numbers (cos + i*sin)
        freq_cis = np.cos(freqs_for_each_token) + 1j * np.sin(freqs_for_each_token)
        
        return freq_cis
    
    def apply_rotary_pos_emb(self, x):
        """Apply rotary position embedding to input"""
        seq_len = x.shape[1]
        
        # Reshape x to complex numbers (treating pairs of dimensions as complex)
        x_complex = x[..., ::2] + 1j * x[..., 1::2]
        
        # Apply rotation
        freq_cis = self.freq_cis[:seq_len, :]
        x_rotated = x_complex * freq_cis[np.newaxis, :, :]
        
        # Convert back to real numbers
        x_out = np.zeros_like(x)
        x_out[..., ::2] = np.real(x_rotated)
        x_out[..., 1::2] = np.imag(x_rotated)
        
        return x_out

# Compare different position encoding schemes
def compare_position_encodings():
    """Compare different positional encoding methods"""
    d_model, seq_len, batch_size = 64, 20, 1
    
    # Create sample input (without position info)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Sinusoidal encoding
    sin_pe = PositionalEncoding(d_model)
    x_sin = sin_pe.forward(x.copy())
    
    # Learned encoding
    learned_pe = LearnedPositionalEmbedding(d_model, seq_len)
    x_learned = learned_pe.forward(x.copy())
    
    # Rotary encoding
    rope = RotaryPositionalEncoding(d_model)
    x_rope = rope.apply_rotary_pos_emb(x.copy())
    
    print("Position Encoding Comparison:")
    print(f"Original: {x.shape}, std: {np.std(x):.3f}")
    print(f"Sinusoidal: {x_sin.shape}, std: {np.std(x_sin):.3f}")
    print(f"Learned: {x_learned.shape}, std: {np.std(x_learned):.3f}")
    print(f"RoPE: {x_rope.shape}, std: {np.std(x_rope):.3f}")

compare_position_encodings()
```

## 🧠 Complete Language Model Architecture

### GPT-Style Decoder-Only Model

**Full GPT Implementation**:
```python
class GPTModel:
    """Complete GPT-style language model implementation"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 d_ff: int = 3072,
                 max_seq_len: int = 1024,
                 dropout_rate: float = 0.1):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Token embeddings
        self.token_embeddings = self._init_embeddings(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            self.transformer_blocks.append(block)
        
        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        
        # Language modeling head
        self.lm_head = self._init_weights((d_model, vocab_size))
        
        # For storing attention patterns during forward pass
        self.attention_patterns = []
    
    def _init_embeddings(self, vocab_size, d_model):
        """Initialize token embeddings"""
        return np.random.randn(vocab_size, d_model) * 0.02
    
    def _init_weights(self, shape):
        """Initialize linear layer weights"""
        return np.random.randn(*shape) / np.sqrt(shape[0])
    
    def _create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0  # True for allowed positions
    
    def embed_tokens(self, input_ids):
        """Convert token IDs to embeddings"""
        batch_size, seq_len = input_ids.shape
        
        # Create embedding matrix for batch
        embeddings = np.zeros((batch_size, seq_len, self.d_model))
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = input_ids[i, j]
                embeddings[i, j] = self.token_embeddings[token_id]
        
        return embeddings
    
    def forward(self, input_ids, training=True, return_attention=False):
        """
        Forward pass through GPT model
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            training: Whether in training mode
            return_attention: Whether to return attention patterns
        
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            attention_patterns: List of attention weights (if requested)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed_tokens(input_ids)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        
        # Apply dropout to embeddings
        if training:
            keep_prob = 1 - self.dropout_rate
            mask = np.random.binomial(1, keep_prob, x.shape)
            x = x * mask / keep_prob
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        attention_patterns = []
        for i, block in enumerate(self.transformer_blocks):
            x, attn_weights = block.forward(x, mask=causal_mask, training=training)
            
            if return_attention:
                attention_patterns.append(attn_weights)
        
        # Final layer normalization
        x = self.final_norm.forward(x, training)
        
        # Language modeling head
        logits = x @ self.lm_head
        
        self.attention_patterns = attention_patterns
        
        if return_attention:
            return logits, attention_patterns
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k most likely tokens
        
        Returns:
            generated_ids: Extended sequence with generated tokens
        """
        generated_ids = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(generated_ids, training=False)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_indices = np.argsort(next_token_logits, axis=-1)[:, -top_k:]
                mask = np.zeros_like(next_token_logits)
                for i in range(next_token_logits.shape[0]):
                    mask[i, top_k_indices[i]] = 1
                next_token_logits = np.where(mask, next_token_logits, -np.inf)
            
            # Apply softmax to get probabilities
            probs = softmax(next_token_logits)
            
            # Sample next token
            next_token_ids = []
            for i in range(probs.shape[0]):
                next_token_id = np.random.choice(self.vocab_size, p=probs[i])
                next_token_ids.append(next_token_id)
            
            next_token_ids = np.array(next_token_ids).reshape(-1, 1)
            
            # Append to sequence
            generated_ids = np.concatenate([generated_ids, next_token_ids], axis=1)
            
            # Stop if we hit max sequence length
            if generated_ids.shape[1] >= self.max_seq_len:
                break
        
        return generated_ids
    
    def get_model_size(self):
        """Calculate total number of parameters"""
        total_params = 0
        
        # Token embeddings
        total_params += self.vocab_size * self.d_model
        
        # Transformer blocks
        for block in self.transformer_blocks:
            # Multi-head attention
            total_params += 4 * (self.d_model * self.d_model)  # Q, K, V, O projections
            
            # Feed-forward network
            total_params += self.d_model * self.d_ff  # First layer
            total_params += self.d_ff * self.d_model  # Second layer
            total_params += self.d_ff + self.d_model  # Biases
            
            # Layer norms
            total_params += 2 * self.d_model  # Gamma parameters
            total_params += 2 * self.d_model  # Beta parameters
        
        # Final layer norm
        total_params += 2 * self.d_model
        
        # Language modeling head  
        total_params += self.d_model * self.vocab_size
        
        return total_params
    
    def visualize_attention_patterns(self, input_text, tokenizer=None):
        """Visualize attention patterns for input text"""
        # This would require a tokenizer to convert text to tokens
        # For now, we'll show the structure
        
        if not self.attention_patterns:
            print("No attention patterns available. Run forward() first.")
            return
        
        num_layers = len(self.attention_patterns)
        num_heads = self.attention_patterns[0].shape[1]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(min(6, num_layers)):
            # Show first head of each layer
            attn_weights = self.attention_patterns[i][0, 0]  # First batch, first head
            
            im = axes[i].imshow(attn_weights, cmap='Blues')
            axes[i].set_title(f'Layer {i+1}, Head 1')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.show()

# Test complete GPT model
def test_gpt_model():
    """Test the complete GPT implementation"""
    # Small model for testing
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 4
    max_seq_len = 64
    
    # Create model
    gpt = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    print(f"Model created with {gpt.get_model_size():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 20
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, attention_patterns = gpt.forward(input_ids, return_attention=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of attention pattern layers: {len(attention_patterns)}")
    
    # Test generation
    generated = gpt.generate(input_ids, max_new_tokens=10, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Visualize attention
    gpt.visualize_attention_patterns("test input")
    
    print("✓ GPT model test passed!")

test_gpt_model()
```

## 📊 Architecture Variations

### Different Transformer Architectures

**Comparison of Major Architectures**:

| Model Family | Architecture | Key Features | Use Cases |
|--------------|-------------|--------------|-----------|
| **GPT** | Decoder-only | Causal attention, autoregressive | Text generation, chat |
| **BERT** | Encoder-only | Bidirectional attention | Classification, NLU |
| **T5** | Encoder-Decoder | Full transformer | Translation, summarization |
| **PaLM** | Decoder-only | Improved scaling, parallel layers | Large-scale generation |
| **LLaMA** | Decoder-only | RMSNorm, SwiGLU, RoPE | Efficient large models |

**Architecture Comparison**:
```python
def compare_architectures():
    """Compare different transformer architectures"""
    
    print("Transformer Architecture Comparison:")
    print("=" * 50)
    
    architectures = {
        'GPT-3': {
            'type': 'Decoder-only',
            'layers': 96,
            'd_model': 12288,
            'heads': 96,
            'parameters': '175B',
            'attention': 'Causal',
            'use_case': 'Generation'
        },
        'BERT-Large': {
            'type': 'Encoder-only', 
            'layers': 24,
            'd_model': 1024,
            'heads': 16,
            'parameters': '340M',
            'attention': 'Bidirectional',
            'use_case': 'Understanding'
        },
        'T5-Large': {
            'type': 'Encoder-Decoder',
            'layers': '24 (12+12)',
            'd_model': 1024,
            'heads': 16,
            'parameters': '770M',
            'attention': 'Full + Causal',
            'use_case': 'Text-to-Text'
        }
    }
    
    for name, specs in architectures.items():
        print(f"\n{name}:")
        for key, value in specs.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

compare_architectures()
```

## ✅ Architecture Understanding Check

Before proceeding, ensure you understand:

1. **Self-Attention**: How queries, keys, and values interact
2. **Multi-Head Attention**: Parallel attention computations
3. **Transformer Blocks**: Layer norm, residual connections, FFN
4. **Position Encoding**: Different methods for position information
5. **Causal Masking**: How autoregressive models work
6. **Model Scaling**: Parameter count and computational complexity

## 🚀 Next Steps

With transformer architecture mastered, continue to:

1. **[Training Process](training.md)** - How LLMs are trained at scale
2. **[Fine-tuning & Adaptation](fine-tuning.md)** - Customizing models for tasks
3. **[Prompt Engineering](prompt-engineering.md)** - Effective LLM interaction

---

*Understanding transformer architecture is crucial for working with LLMs effectively. This foundation enables you to customize, fine-tune, and build agent systems on top of these powerful models.*
