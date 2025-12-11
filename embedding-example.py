import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=1000, embedding_dim=768)

token_ids = torch.tensor([5, 27, 999])
vectors = embedding(token_ids)

print(vectors)