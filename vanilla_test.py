import time
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import datasets
from tqdm import tqdm
import gc
import csv
import random

target_model_names = ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"]
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

def vanilla_inference(target_model_name, prompts, max_new_tokens=max_new_tokens):
    total_inference_time = 0
    total_token_length = 0
    print("Vanilla Inference")
    print(f"Model: {target_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
    )
    target_model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, 
        do_sample=True,
        num_beams=1,
        num_return_sequences=1)

    target_model.to('cuda')

    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            start_time = time.time()
            output = target_model.generate(input_ids, generation_config=generation_config)
            end_time = time.time()
        total_inference_time += end_time - start_time
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        print(response)

    tokens_per_second = total_token_length / total_inference_time
    print(f"Tokens per second (vanilla): {tokens_per_second}")
    
    # Clear memory
    del target_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return tokens_per_second

csv_data = {}
for target_model in target_model_names:
    token_per_second = vanilla_inference(target_model, prompts)
    csv_data[target_model] = token_per_second

with open('vanilla_inference_alpaca_temp_0.csv', mode='w') as file:
    writer = csv.writer(file)
    for key, value in csv_data.items():
        writer.writerow([key, value])
