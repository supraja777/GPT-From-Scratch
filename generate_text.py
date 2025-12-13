import torch
import tiktoken
from gpt_model import GPTModel
from gpt_config_2 import GPT_CONFIG_124M

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature = 0.0, top_K = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        if top_K is not None:
            top_logits, _ = torch.topK(logits, top_K)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensot(float('-inf')).to(logits.device),
                logits
            )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim = -1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx


# start_context = "Hello, I am"
# tokenizer = tiktoken.get_encoding("gpt2")
# encoded = tokenizer.encode(start_context)
# print("Encoded: ", encoded)
# encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
# print("encoded_tensor.shape : ", encoded_tensor.shape)

# model = GPTModel(GPT_CONFIG_124M)

# #model.eval()
# out = generate_text_simple(
#     model=model,
#     idx=encoded_tensor,
#     max_new_tokens=6,
#     context_size=GPT_CONFIG_124M["context_length"]
# )

# print("Output : ", out)
# print("Output length: ", len(out[0]))

# # Decoding the text back
# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)