import time
import torch
from transformers import AutoTokenizer
from inference.fork_shape_tree_attn import *
from inference.strategies import *
from inference.decision_models import *
from model.llama_tree_attn import LlamaForCausalLM
import datasets
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import csv

target_model_name = "meta-llama/Llama-2-7b-chat-hf"
draft_model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

dataset = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
prompts = [" ".join(example['prompt']) for example in dataset]

max_new_tokens = 200
draft_model_temp = 0
target_model_temp = 0

def dynasd_inference(target_model_name, draft_model_name, prompts, draft_model_temp=draft_model_temp, target_model_temp=target_model_temp):
    widths = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    depth = 5
    
    if draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and target_model_name == 'lmsys/vicuna-7b-v1.5':
        file_name = "tinyllama_vicuna_7b"
    elif draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and target_model_name == 'meta-llama/Llama-2-7b-chat-hf':
        file_name = "tinyllama_llama2_7b"
    elif draft_model_name == 'JackFram/llama-68m' and target_model_name == 'lmsys/vicuna-7b-v1.5':
        file_name = "llama_68m_vicuna_7b"
    elif draft_model_name == 'JackFram/llama-68m' and target_model_name == 'meta-llama/Llama-2-7b-chat-hf':
        file_name = "llama_68m_llama2_7b"
    
    if draft_model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
        decision_model = DecisionModelV1_Tinyllama().cuda()
    elif draft_model_name == 'JackFram/llama-68m':
        decision_model = DecisionModelV1().cuda()

    print("DynaSD Inference")
    
    decision_model_path = f'/home/iasl-transformers/DynaSD/decision_model/soft_{file_name}.pt'
    draft_model = LlamaForCausalLM.from_pretrained(draft_model_name, torch_dtype=torch.float32).to('cuda')
    target_model = LlamaForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.bfloat16).to('cuda')
    draft_model.eval()
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    acc_rates = []
    speeds = []

    for width in widths:
        strategy = NewTreeStrategy(
            draft_model=draft_model,
            target_model=target_model,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            greedy=True,
            using_decision_model=False,
            decision_model=decision_model,
            decision_model_path=decision_model_path,
            decision_threshold=0.4,
            config_depth=depth,
            config_width=width,
            draft_model_temp=draft_model_temp,
            target_model_temp=target_model_temp,
            soft_label=True
        )

        strategy.max_config = strategy.generate_fork_config(width=width, depth=depth)
        strategy.tree_attn_self_mask = get_tree_attn_self_mask(strategy.max_config).to("cuda")
        strategy.max_draft_len = len(strategy.max_config) - 1
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
        acc_rates.append(acc_rate)
        speeds.append(speed)

    torch.cuda.empty_cache()
    del draft_model
    del target_model
    del tokenizer
    gc.collect()

    return widths, acc_rates, speeds

# widths, acc_rates, speeds = dynasd_inference(target_model_name=target_model_name, draft_model_name=draft_model_name, prompts=prompts)
# with open('/home/iasl-transformers/DynaSD/results/dynasd_acc_rate_speed_vs_width_no_decision_model_large_model.csv', mode='w') as file:
#     writer = csv.writer(file)
#     writer.writerow([target_model_name, draft_model_name, widths, acc_rates, speeds])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Small draft model
with open('/home/iasl-transformers/DynaSD/results/dynasd_acc_rate_speed_vs_width.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        target_model = row[0]
        draft_model = row[1]
        if target_model == 'meta-llama/Llama-2-7b-chat-hf' and draft_model == 'JackFram/llama-68m':
            widths = row[2].strip('][').split(', ')
            widths = list(map(int, widths))
            speeds = row[4].strip('][').split(', ')
            speeds = list(map(float, speeds))
            axs[0].plot(widths, speeds, label=f"With Decision Model")
            break

with open('/home/iasl-transformers/DynaSD/results/dynasd_acc_rate_speed_vs_width_no_decision_model.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        target_model = row[0]
        draft_model = row[1]
        if target_model == 'meta-llama/Llama-2-7b-chat-hf' and draft_model == 'JackFram/llama-68m':
            widths = row[2].strip('][').split(', ')
            widths = list(map(int, widths))
            speeds = row[4].strip('][').split(', ')
            speeds = list(map(float, speeds))
            axs[0].plot(widths, speeds, label=f"Without Decision Model") 
            break

axs[0].set_xlabel('Width')
axs[0].set_ylabel('Generation Speed (tokens/s)')
axs[0].set_ylim(bottom=100, top=180)
axs[0].legend()
axs[0].grid(True)

# Plot 2: Large draft model
with open('/home/iasl-transformers/DynaSD/results/dynasd_acc_rate_speed_vs_width.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        target_model = row[0]
        draft_model = row[1]
        if target_model == 'meta-llama/Llama-2-7b-chat-hf' and draft_model == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
            widths = row[2].strip('][').split(', ')
            widths = list(map(int, widths))
            speeds = row[4].strip('][').split(', ')
            speeds = list(map(float, speeds))
            print(speeds)
            axs[1].plot(widths, speeds, label=f"With Decision Model")
            break

with open('/home/iasl-transformers/DynaSD/results/dynasd_acc_rate_speed_vs_width_no_decision_model.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        target_model = row[0]
        draft_model = row[1]
        if target_model == 'meta-llama/Llama-2-7b-chat-hf' and draft_model == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
            widths = row[2].strip('][').split(', ')
            widths = list(map(int, widths))
            speeds = row[4].strip('][').split(', ')
            speeds = list(map(float, speeds))
            print(speeds)
            axs[1].plot(widths, speeds, label=f"Without Decision Model")
            break

axs[1].set_xlabel('Width')
axs[1].set_ylabel('Generation Speed (tokens/s)')
axs[1].set_ylim(bottom=70, top=85)
axs[1].legend()
axs[1].grid(True)

# Save and show the plots
plt.tight_layout()
plt.savefig('dynasd_speed_vs_width_decision_model.png')
plt.show()
