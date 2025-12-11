import torch.nn as nn
import torch

inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


class SelfAttention_v1(nn.Module):
   def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
      super().__init__()
      self.d_out = d_out
      self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
      self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
      self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
      self.dropout = nn.Dropout(dropout)
      self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )      
   
   def forward(self, x):
      b, num_tokens, d_in = x.shape  
      keys = self.W_key(x)
      queries = self.W_query(x)
      values = self.W_value(x)

      attn_scores = queries @ keys.transpose(1, 2)
      # attn_weights = torch.softmax(
      #    attn_scores / keys.shape[-1] ** 0.5, dim = -1
      # )

      # context_len = attn_scores.shape[0]
      # mask_simple = torch.tril(torch.ones(context_len, context_len))
      # mask_simple = attn_weights * mask_simple
      # print(mask_simple)

      # mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
      attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
      attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
      attn_weights = self.dropout(attn_weights) # Dropping 50% of the attention weights

      context_vec = attn_weights @ values
      return context_vec

# torch.manual_seed(123)
# batch = torch.stack((inputs, inputs), dim = 0)
# context_length = inputs.shape[0]
# d_in = inputs.shape[1]
# d_out = 2
# sa_v1 = SelfAttention_v1(d_in, d_out, context_length, 0.0)
# print(sa_v1(batch))
# print("-------------------")