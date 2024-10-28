import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.speculative_decoding.speculative_sampling import greedy_speculative_sampling, speculative_sampling
import datasets
from tqdm import tqdm
import gc
import csv
import random

target_model_names = ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"]
draft_model_names = ['TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'JackFram/llama-68m']
dataset_names = ["mt_bench", "alpaca", "trivia_qa"]
sampling_methods = ["probabilistic"]

max_new_tokens = 200

def speculative_decoding_inference(target_model_name, draft_model_name, prompts, max_new_tokens=max_new_tokens):
    print("Naive Speculative Decoding Inference")
    total_inference_time = 0
    total_token_length = 0
    total_acceptance_count = 0
    total_draft_token_count = 0

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
            output, acc_count, draft_count = speculative_sampling(
                prefix=input_ids,
                gamma=4,
                temperature=0.7,
                approx_model=draft_model,
                target_model=target_model, 
                max_len=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id)
            end_time = time.time()
        total_inference_time += end_time - start_time
        total_acceptance_count += acc_count
        total_draft_token_count += draft_count
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        torch.cuda.empty_cache()
        print(f"Output length: {output.shape[1] - input_ids.shape[1]}")
        print(response)
    tokens_per_second = total_token_length / total_inference_time
    acc_rate = total_acceptance_count / total_draft_token_count
    torch.cuda.empty_cache()
    del draft_model
    del target_model
    del tokenizer
    gc.collect()
    return tokens_per_second, acc_rate

for dataset_name in dataset_names:
    prompts = []
    if dataset_name == "mt_bench":
        dataset = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [" ".join(example['prompt']) for example in dataset]
    elif dataset_name == "alpaca":
        dataset = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
        prompts = []
        for example in dataset:
            if len(example['input']) != 0:
                prompt = "Question: " + " ".join([example['instruction'], example["input"]]).strip() + " Answer: "
                prompts.append(prompt)
            else:
                prompt = "Answer the following question: " + example['instruction']
                prompts.append(prompt)
        prompts = random.sample(prompts, 250)
    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="test")
        prompts = []
        for example in dataset:
            prompt = "Answer the following question: " + example['question'] + " Write your answer here: "
            prompts.append(prompt)
        prompts = random.sample(prompts, 250)
    for sampling_method in sampling_methods:
        csv_data = {}
        for target_model in target_model_names:
            for draft_model in draft_model_names:
                print(f"Dataset: {dataset_name}")
                print(f"Target model: {target_model}, Draft model: {draft_model}")
                print(f"Sampling method: {sampling_method}")
                speed, acc_rate = speculative_decoding_inference(
                    target_model_name=target_model, 
                    draft_model_name=draft_model, 
                    prompts=prompts,
                    max_new_tokens=max_new_tokens)
                csv_data[(draft_model, target_model)] = [speed, acc_rate]
                print(f"Speed: {speed}")
                print(f"Acceptance Rate: {acc_rate}")
        with open(f'sd_inference_{dataset_name}_{sampling_method}_temp_0.7_gamma_4.csv', mode='w') as file:
            writer = csv.writer(file)
            for key, value in csv_data.items():
                writer.writerow([key[0], key[1], value[0], value[1]])