import time
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from speculative_decoding.speculative_sampling import greedy_speculative_sampling, speculative_sampling
import datasets
from tqdm import tqdm
import gc
import csv
import random

target_model_names = ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"]
draft_model_names = ['TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'JackFram/llama-68m']
# dataset = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
# prompts = [" ".join(example['prompt']) for example in dataset]

dataset = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
prompts = []
for example in dataset:
    if len(example['input']) != 0:
        prompt = "Question: " + " ".join([example['instruction'], example["input"]]).strip() + " Answer: "
        prompts.append(prompt)
    else:
        prompt = "Answer the following question: " + example['instruction']
        prompts.append(prompt)
prompts = random.sample(prompts, 500)

max_new_tokens = 200
use_greedy = False

def speculative_decoding_inference(target_model_name, draft_model_name, prompts, max_new_tokens=max_new_tokens):
    print("Naive Speculative Decoding Inference")
    total_inference_time = 0
    total_token_length = 0

    print(f"Draft Model: {draft_model_name}")
    print(f"Target Model: {target_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float32,
    ).to('cuda')
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    draft_model.eval()
    target_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            start_time = time.time()
            if use_greedy:
                output, acc_rate = greedy_speculative_sampling(
                    prefix=input_ids,
                    approx_model=draft_model,
                    target_model=target_model,
                    max_len=max_new_tokens)
            else:
                output, acc_rate = speculative_sampling(
                    prefix=input_ids,
                    temperature=1,
                    approx_model=draft_model,
                    target_model=target_model, 
                    max_len=max_new_tokens)
            end_time = time.time()
        total_inference_time += end_time - start_time
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        print(response)
    tokens_per_second = total_token_length / total_inference_time
    torch.cuda.empty_cache()
    del draft_model
    del target_model
    del tokenizer
    gc.collect()
    return tokens_per_second, acc_rate

csv_data = {}
for target_model in target_model_names:
    for draft_model in draft_model_names:
        token_per_second, acc_rate = speculative_decoding_inference(
            target_model_name=target_model,
            draft_model_name=draft_model,
            prompts=prompts)
        csv_data[(draft_model, target_model)] = [token_per_second, acc_rate]
        print(f"Tokens per second: {token_per_second}")

with open('sd_inference_alpaca_temp_1.csv', mode='w') as file:
    writer = csv.writer(file)
    for key, value in csv_data.items():
        writer.writerow([key[0], key[1], value[0], value[1]])
