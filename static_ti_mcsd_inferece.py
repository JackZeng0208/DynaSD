import time
import torch
from inference.target_initialized_mcsd import *
from transformers import AutoTokenizer
# from inference.fork_shape_tree_attn import *
# from inference.strategies import *
# from inference.decision_models import *
from model.llama_tree_attn import LlamaForCausalLM
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
def dynasd_inference(target_model_name, 
                     draft_model_name, 
                     prompts, 
                     temperature, 
                     max_new_tokens=max_new_tokens, 
):
    mcsd_config = (2,4,4,1,1)
    draft_model = LlamaForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    target_model = LlamaForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    draft_model.eval()
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    mcsd_strategy = TargetInitializedtMCSD(
        draft_model,
        target_model,
        mcsd_config,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature
    )

    acc_count = 0
    total_generated_draft_tokens = 0
    total_generated_tokens = 0
    
    start_time = time.time()
    with torch.no_grad():
        for prompt_text in tqdm(prompts):
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            output, stats = mcsd_strategy.generation_loop(input_ids=input_ids)
            # print(tokenizer.batch_decode(output, skip_special_tokens=True))
            total_generated_tokens += output.shape[1] - input_ids.shape[1]
            acc_count += stats["ground_acceptance_count"].item()
            total_generated_draft_tokens +=stats["total_generated_draft_tokens"]
    end_time = time.time()
    run_time = end_time - start_time
    acc_rate = acc_count / total_generated_draft_tokens
    speed = total_generated_tokens / run_time
    torch.cuda.empty_cache()
    del draft_model
    del target_model
    del tokenizer
    gc.collect()
    return speed, acc_rate

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
        if sampling_method == "greedy":
            temperature = 0
        elif sampling_method == "probabilistic":
            temperature = 0.7
        for target_model in target_model_names:
            for draft_model in draft_model_names:
                print(f"Dataset: {dataset_name}")
                print(f"Target model: {target_model}, Draft model: {draft_model}")
                print(f"Sampling method: {sampling_method}")
                speed, acc_rate = dynasd_inference(
                    target_model_name=target_model, 
                    draft_model_name=draft_model, 
                    prompts=prompts,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                    )
                csv_data[(draft_model, target_model)] = [speed, acc_rate]
                print(f"Speed: {speed}")
                print(f"Acceptance Rate: {acc_rate}")
        with open(f'dynasd_inference_{dataset_name}_{sampling_method}.csv', mode='w') as file:
            writer = csv.writer(file)
            for key, value in csv_data.items():
                writer.writerow([key[0], key[1], value[0], value[1]])
