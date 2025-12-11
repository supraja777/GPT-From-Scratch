GPT_CONFIG_124M = {
    "vocab_size": 50257,   # number of words used by BPE tokenizer
    "context_length": 1024, # Maximum number of input tokens the model can handle via positional emb
    "emb_dim": 768, # Represents the embedding size, transforming each token into a 768 dimensional space
    "n_heads": 12, # Count of attention heads in multi-head attention mechanism
    "n_layers": 12, # Number of transformer blocks
    "drop_rate": 0.1, 
    "qkv_bias": False
}

