# from speculative_sampling import* 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import tqdm
# import time
# import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


# draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name,torch_dtype=torch.float32)
# target_model = AutoModelForCausalLM.from_pretrained(target_model_name,torch_dtype=torch.float16)

@torch.no_grad()
def experiment():
    model_name = {'apple270m': {'name': 'apple/OpenELM-270M-Instruct', 'datatype': torch.bfloat16},
              'apple3b': {'name': 'apple/OpenELM-3B-Instruct', 'datatype': torch.bfloat16},
              'llamatiny': {'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'datatype': torch.bfloat16},
              'llama68m': {'name': 'JackFram/llama-68m', 'datatype': torch.bfloat16},
              'llama160m': {'name': 'JackFram/llama-160m', 'datatype': torch.bfloat16},
              'llama7b': {'name': 'meta-llama/Llama-2-7b-chat-hf', 'datatype': torch.bfloat16}}

    target_model_name = model_name['llama7b']['name']
    draft_model_name = model_name['llama68m']['name']
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",padding_side = "left")
    model = AutoModelForCausalLM.from_pretrained(draft_model_name,torch_dtype=torch.float32)
    # batch = ['hello world,', 'once upon a time,','tell me something about UCI']
    batch = ['hello world']
    # batch = 'once upon a time,'
    tokenizer.pad_token = tokenizer.eos_token#({'pad_token': '[PAD]'})
    # tokenizer.eos_token = '[PAD]'
    batched_inputs = tokenizer(batch, padding=True,return_tensors='pt')
    print(f"batched_input is {batched_inputs}")
    print(tokenizer.special_tokens_map)
    # model.to("cuda:0") 
    # Generate text using the LLAMA2 model


    output_ids = model(**batched_inputs)
    print('shape of kvcache:', np.shape(output_ids.past_key_values)) 

experiment()
# Decode the generated output
# output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
# print(output_text)