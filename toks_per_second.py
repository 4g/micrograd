import random
from time import sleep
from transformers import AutoTokenizer, LlamaTokenizer

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-chat-fp16")

text = open('toks_per_second.py').read()
tokens = tokenizer.encode(text)
toks_per_second = 8.8

for i in range(len(tokens)):
    sleep(1./toks_per_second)
    print(f"{tokenizer.decode(tokens[i], skip_special_tokens=True)}", end='', flush=True)
