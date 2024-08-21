import time
import torch
from transformers import AutoTokenizer
from inference.fork_shape_tree_attn import *
from inference.strategies import *
from decision_models import *
from model.llama_tree_attn import LlamaForCausalLM
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
draft_model_temp = 1
target_model_temp = 1
def dynasd_inference(target_model_name, draft_model_name, prompts, draft_model_temp=draft_model_temp, target_model_temp=target_model_temp):
    width = 6
    if draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and target_model_name == 'lmsys/vicuna-7b-v1.5':
        depth = 4
        decision_threshold = 0.6
        file_name = "tinyllama_vicuna_7b"
    elif draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and target_model_name == 'meta-llama/Llama-2-7b-chat-hf':
        depth = 5
        decision_threshold = 0.6
        file_name = "tinyllama_llama2_7b"
    elif draft_model_name == 'JackFram/llama-68m' and target_model_name == 'lmsys/vicuna-7b-v1.5':
        depth = 6
        decision_threshold = 0.5
        file_name = "llama_68m_vicuna_7b"
    elif draft_model_name == 'JackFram/llama-68m' and target_model_name == 'meta-llama/Llama-2-7b-chat-hf':
        depth = 6
        decision_threshold = 0.5
        file_name = "llama_68m_llama2_7b"
    if draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
        decision_model = DecisionModelV1_Tinyllama().cuda()
    elif draft_model_name == 'JackFram/llama-68m':
        decision_model = DecisionModelV1().cuda()
    print("DynaSD Inference")
    
    total_stats = {'token/second': 0.0,"verification_time": 0.0,
                'acceptance_rate': 0.0, "draft_generation_time": 0.0,
                "ground_acceptance_count": 0, "total_generation_round": 0,
                "decision_model_time": 0.0, "decision_acceptance_count": 0}
    
    decision_model_path = f'/home/iasl-transformers/DynaSD/decision_model/soft_{file_name}.pt'
    draft_model = LlamaForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float32,
    ).to('cuda')
    target_model = LlamaForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    draft_model.eval()
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    strategy = NewTreeStrategy(
        draft_model=draft_model,
        target_model=target_model,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        greedy=False,
        using_decision_model=True,
        decision_model=decision_model,
        decision_model_path=decision_model_path,
        decision_threshold= decision_threshold,
        config_depth= depth,
        config_width= width,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        soft_label=True
    )

    strategy.max_config = strategy.generate_fork_config(width = width,depth = depth)
    strategy.tree_attn_self_mask = get_tree_attn_self_mask(strategy.max_config).to("cuda")
    strategy.max_draft_len = len(strategy.max_config)-1

    acc_count = 0
    total_generated_draft_tokens = 0
    total_generated_tokens = 0
    
    start_time = time.time()
    with torch.no_grad():
        for prompt_text in tqdm(prompts):
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            output, stats = strategy.generation_loop(input_ids=input_ids)
            print(tokenizer.batch_decode(output, skip_special_tokens=True))
            total_generated_tokens += output.shape[1] - input_ids.shape[1]
            acc_count += stats["ground_acceptance_count"].item()
            total_generated_draft_tokens += stats["decision_acceptance_count"]
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

csv_data = {}
for target_model in target_model_names:
    for draft_model in draft_model_names:
        speed, acc_rate = dynasd_inference(target_model_name=target_model, draft_model_name=draft_model, prompts=prompts)
        csv_data[(draft_model, target_model)] = [speed, acc_rate]
        print(f"Speed: {speed}")
        print(f"Acceptance Rate: {acc_rate}")

with open('dynasd_inference_alpaca_temp_1.csv', mode='w') as file:
    writer = csv.writer(file)
    for key, value in csv_data.items():
        writer.writerow([key[0], key[1], value[0], value[1]])
