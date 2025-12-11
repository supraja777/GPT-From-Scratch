import torch 
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Embedding is a learnable lookup table that converts token IDs to dense vectors
        # These embeddings are continuous, dense and learned during training
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
