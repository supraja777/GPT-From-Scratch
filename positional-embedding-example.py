import torch 
import torch.nn as nn

seq_len = 4
emb_dim = 3

pos_emb = nn.Embedding(seq_len, emb_dim)
pos_ids = torch.arange(seq_len)
pos_vectors = pos_emb(pos_ids[1])

print(pos_vectors)