import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from tqdm import tqdm
import gc
import csv
import random

target_model_names = ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"]
dataset_names = ["mt_bench", "alpaca", "trivia_qa"]
sampling_methods = ["probabilistic"]

max_new_tokens = 200

def vanilla_inference(target_model_name, prompts, max_new_tokens=max_new_tokens, use_greedy=True):
    total_inference_time = 0
    total_token_length = 0
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
    )
    target_model.eval()
    target_model.to('cuda')
    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            start_time = time.time()
            output = target_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            end_time = time.time()
        total_inference_time += end_time - start_time
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        torch.cuda.empty_cache()
        print(response)

    tokens_per_second = total_token_length / total_inference_time
    print(f"Tokens per second (vanilla): {tokens_per_second}")
    
    # Clear memory
    del target_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return tokens_per_second

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
        use_greedy = False
        # for target_model in target_model_names:
        for target_model in target_model_names:
            print(f"Dataset: {dataset_name}")
            print(f"Model: {target_model}")
            print(f"Sampling method: {sampling_method}")
            token_per_second = vanilla_inference(
                target_model_name=target_model, 
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                use_greedy=use_greedy)
            csv_data[target_model] = token_per_second
            print(f"Speed: {token_per_second}")
        with open(f'vanilla_inference_{dataset_name}_{sampling_method}_temp_0.7.csv', mode='w') as file:
            writer = csv.writer(file)
            for key, value in csv_data.items():
                writer.writerow([key, value])