from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from gpt_config import BASE_CONFIG
from load_weights_into_gpt import load_weights_into_gpt
from generate_text import generate_text_simple
from encoding_and_decoding_tokens import text_to_token_ids, token_ids_to_text
import tiktoken

model_size = "124M"
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text1 = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text1, tokenizer),
    max_new_tokens = 15,
    context_size = BASE_CONFIG["context-length"]
)

print(token_ids_to_text(token_ids, tokenizer))