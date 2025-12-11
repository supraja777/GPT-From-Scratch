from self_attention_v1 import SelfAttention_v1
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttention_v1(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(d_out*num_heads, d_out)
    
    def forward(self, x):
        head_outputs = []
        # for idx, head in enumerate(self.heads):
        #     out = head(x)
        #     print(f"Head {idx + 1} output: \n", out)
        #     head_outputs.append(out)
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(context_vec)
        # return torch.cat([head(x) for head in self.heads], dim = -1)

# inputs = torch.tensor(
#    [[0.43, 0.15, 0.89], # Your     (x^1)
#    [0.55, 0.87, 0.66], # journey  (x^2)
#    [0.57, 0.85, 0.64], # starts   (x^3)
#    [0.22, 0.58, 0.33], # with     (x^4)
#    [0.77, 0.25, 0.10], # one      (x^5)
#    [0.05, 0.80, 0.55]] # step     (x^6)
# )

# torch.manual_seed(123)
# batch = torch.stack((inputs, inputs), dim = 0)
# context_length = batch.shape[1]
# d_in, d_out = 3, 3
# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=6)
# context_vecs = mha(batch)
# print(context_vecs)