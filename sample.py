import os
import argparse
import tiktoken
import time
import json

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

model_weights_path = os.path.join(out_dir, out_dir + '.npz')
model_config_path = os.path.join(out_dir, out_dir + '.json')

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    with open(model_config_path, "r") as f:
        config_args = json.load(f)

    config = GPTConfig(**config_args)
    model = GPT(config)

    weights = mx.load(model_weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded GPT-2 with {nparams / 1e6:.3f} M parameters")

elif init_from.startswith('gpt2'):
    # init from OpenAI GPT-2 checkpoint
    print(f"Initializing from OpenAI GPT-2 checkpoint: {init_from}")
    
    # map init_from to model size
    model_size_map = {
        'gpt2': 124,         # 124M params
        'gpt2-medium': 350,  # 350M params  
        'gpt2-large': 774,   # 774M params
        'gpt2-xl': 1558,     # 1.5B params
    }
    
    if init_from not in model_size_map:
        raise ValueError(f"Unknown GPT-2 model: {init_from}. Available: {list(model_size_map.keys())}")
    
    # create a from-scratch initialized minGPT model
    config_args = {
        'n_layer': 12, 'n_head': 12, 'n_embd': 768,  # gpt2 default
        'block_size': 1024,
        'bias': True,
        'vocab_size': 50257,  # openai's model vocabulary
        'dropout': 0.0,
    }
    
    # override defaults based on model size
    if init_from == 'gpt2-medium':
        config_args.update(dict(n_layer=24, n_head=16, n_embd=1024))
    elif init_from == 'gpt2-large': 
        config_args.update(dict(n_layer=36, n_head=20, n_embd=1280))
    elif init_from == 'gpt2-xl':
        config_args.update(dict(n_layer=48, n_head=25, n_embd=1600))
    
    config = GPTConfig(**config_args)
    model = GPT(config)
    
    # For now, we'll use random weights. In a full implementation, you would:
    # 1. Download the OpenAI GPT-2 checkpoint
    # 2. Convert PyTorch state_dict to MLX format  
    # 3. Load the converted weights
    print(f"Note: Using randomly initialized {init_from} model. For actual OpenAI weights, manual conversion is needed.")
    
    mx.eval(model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded {init_from} with {nparams / 1e6:.3f} M parameters")

# ok let's assume gpt-2 encodings by default
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (mx.array([start_ids], dtype=mx.uint32))

# run generation
start = time.time()
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
end = time.time()
