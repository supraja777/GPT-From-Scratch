# The main idea behind layer normalization is to adjust the activations (outputs) of a neural
# network to have mean 0 and variance of 1 (unit variance)

# It is applied before and after the multihead attention module
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # This layer normalization acts on the last dimension i.e. the embedding dimension
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
ln = LayerNorm(emb_dim=5)
batch_example = torch.randn(2, 5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim = -1, keepdim = True)
var = out_ln.var(dim = -1, keepdim = True, unbiased=False)

print("Mean: \n", mean)
print("Variance: \n", var)
