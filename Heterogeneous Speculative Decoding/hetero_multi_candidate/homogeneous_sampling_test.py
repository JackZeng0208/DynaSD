# first test on reject sampling, then, 

# then test on typical acceptance

import argparse
import json
import logging
import time
from typing import Literal, Tuple

import torch
from inference.generate import SpeculativeGenerator
from model.llama_tree_attn import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from inference.mcstrategy import TreeStrategy


draft_model_path = 'JackFram/llama-68m'
# target_model_path = 'JackFram/llama-160m'
target_model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# target_model_path = 'meta-llama/Llama-2-7b-chat-hf'
draft_model = LlamaForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.float16,
        # load_in_8bit = True,
        device_map=0,
        use_flash_attention_2= False,
    )

target_model = LlamaForCausalLM.from_pretrained(
        target_model_path,
        # torch_dtype=torch.float16,
        load_in_8bit = True,
        device_map=0,
        use_flash_attention_2= False,
    )



sentence_list = ["Write a story set in a post-apocalyptic world where humans are struggling to survive.",
"Describe a magical forest where every tree has a unique power.",
"Imagine a future where artificial intelligence has surpassed human intelligence.",
"Create a dialogue between a time traveler and a historical figure.",
"Describe a society where emotions are considered a form of currency.",
"Write a poem about the beauty and power of the ocean.",
"Invent a new planet with its own ecosystem and inhabitants.",
"Describe a day in the life of a character who can teleport at will.",
"Write a short story about a secret society hiding in plain sight.",
"Imagine a world where dreams are tangible and can be bought and sold.",
"Once upon a time"]



k_config_e = (2,2,1)
draft_model_temp=1
target_model_temp=1
max_new_tokens = 200

acceptance_count = 0 #+= output.acceptance_count
draft_token_count = 0#+= output.draft_token_count
invocation_count = 0

tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
generator = SpeculativeGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config_e,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=False,
        speculative_sampling=True,
        use_origin= False
    )

draft_model.eval()
target_model.eval()
for i in tqdm(range(len(sentence_list))):
    
    prompt = tokenizer(sentence_list[i],return_tensors="pt").to("cuda:0")
    input_ids = prompt.input_ids

    start_time = time.time()
    output = generator.generate(input_ids)
    end_time = time.time()
    print(f'output is {tokenizer.batch_decode(output.sequences)}')

    acceptance_count += output.acceptance_count
    draft_token_count += output.draft_token_count
    invocation_count += output.invocation_count
    run_time = end_time - start_time
    break
latency = run_time / (acceptance_count + invocation_count)
acceptance_rate = acceptance_count / draft_token_count
block_efficiency = 1 + acceptance_count / invocation_count
print(f'target model use sampling: {target_model_temp == 1}')
print(f'draft model use sampling: {draft_model_temp == 1}')
print(f'k-config is {k_config_e}')
print(f'target model is {target_model_path}')
print(f'draft model is {draft_model_path}')
# print(f'accepted count is {output.acceptance_count}')
# print(f'draft_token_count is {draft_token_count}')
# print(f'invocation count is {invocation_count}')
print(f"Running time: {run_time} s")
print(f"Token latency: {latency * 1000} ms")
print(f"Acceptance rate: {acceptance_rate}")
print(f"Block efficiency: {block_efficiency}")
print(f'token per second is {max_new_tokens/run_time}')



## sanity check
# draft_model_path = 'JackFram/llama-68m'
# # target_model_path = 'JackFram/llama-160m'
# target_model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# # target_model_path = 'meta-llama/Llama-2-7b-chat-hf'
# draft_model = AutoModelForCausalLM.from_pretrained(
#         target_model_path,
#         torch_dtype=torch.float16,
#         # load_in_8bit = True,
#         device_map=0,
#         use_flash_attention_2= False,
#     )

# target_model = AutoModelForCausalLM.from_pretrained(
#         target_model_path,
#         torch_dtype=torch.float16,
#         # load_in_8bit = True,
#         device_map=0,
#         use_flash_attention_2= False,
#     )
# tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
# prompt = tokenizer(sentence_list[0],return_tensors="pt").to("cuda:0")
# input_ids = prompt.input_ids
# max_new_tokens = 200
# start_time = time.time()
# output = target_model.generate(input_ids,assistant_model =draft_model,max_new_tokens =max_new_tokens )
# end_time = time.time()
# print(f'output is {tokenizer.batch_decode(output)}')
# print(f'number / token is {max_new_tokens/(end_time - start_time)}')
