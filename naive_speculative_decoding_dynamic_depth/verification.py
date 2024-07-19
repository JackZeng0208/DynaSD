import time
import torch
from inference.generate import SpeculativeGenerator
from model.llama_tree_attn import LlamaForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
from fork_shape_tree_attn import *
from inference.strategies import *

draft_model_name = 'JackFram/llama-68m'
# target_model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
target_model_name = "meta-llama/Llama-2-7b-chat-hf"

draft_model = LlamaForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype=torch.float32,
    device_map=0,
)

target_model = LlamaForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

draft_model.eval()
target_model.eval()

tokenizer = AutoTokenizer.from_pretrained(target_model_name)
k_config = (5, 1, 1, 1, 1, 1)
max_new_tokens = 200

# Draft model temperature
draft_model_temp = 0
# Target model temperature
target_model_temp = 0

replacement = False
speculative_sampling = True
tree_attn = True

generator = SpeculativeGenerator(
    draft_model,
    target_model,
    eos_token_id=tokenizer.eos_token_id,
    k_config=k_config,
    max_new_tokens=max_new_tokens,
    draft_model_temp=draft_model_temp,
    target_model_temp=target_model_temp,
    replacement=replacement,
    speculative_sampling=speculative_sampling,
    tree_attn=tree_attn,
)

strategy = NewTreeStrategy(
    draft_model=draft_model,
    target_model=target_model,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_new_tokens,
    greedy=False,
    using_decision_model=True,
    decision_threshold= 0.8,
    config_dpeth= 3,
    
)

acceptance_count = 0
draft_token_count = 0
invocation_count = 0
total_tokens = 0

prompts = [
    "once upon a time",
    "I am a computer program.",
    "I am a chatbot.",
    "I am a language model.",
    "I am a transformer language model.",
    "I am a GPT-based language model.",
    "I am an AI assistant.",
    "I am a conversational AI.",
    "I am a chatbot for humans.",
    "I am a program that writes like a human."
]

iterator = range(len(prompts))

# start_time = time.time()
# with torch.no_grad():
#     for sample_idx in tqdm(iterator):
#         prompt_text = prompts[sample_idx]
#         inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
#         input_ids = inputs.input_ids
#         output = generator.generate(input_ids)
#         output_text = tokenizer.batch_decode(
#             output.sequences, skip_special_tokens=True)
#         print(output_text)
#         total_tokens += output.sequences.shape[1] - input_ids.shape[1]
#         acceptance_count += output.acceptance_count
#         draft_token_count += output.draft_token_count
#         invocation_count += output.invocation_count
# end_time = time.time()
# run_time = end_time - start_time
# latency = run_time / (acceptance_count + invocation_count)
# acceptance_rate = acceptance_count / draft_token_count
# block_efficiency = 1 + acceptance_count / invocation_count

# print()
# print(f"Results for MCSD:")
# print(f"Run Time: {run_time:.2f}s")
# print(f"Latency: {latency*1000:.2f}s")
# print(f"Acceptance Rate: {acceptance_rate:.2f}")
# print(f"Tokens/s: {total_tokens/(run_time) :.2f}")
# print()


total_stats = {'token/second': 0.0,"verification_time": 0.0,
               'acceptance_rate': 0.0, "draft_generation_time": 0.0,
               "ground_acceptance_count": 0, "total_generation_round": 0,
               "decision_model_time": 0.0, "decision_acceptance_count": 0}
with torch.no_grad():
    start = time.time()
    for sample_idx in tqdm(iterator):
        prompt_text = prompts[sample_idx]
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        output, stats = strategy.generation_loop(input_ids=input_ids)
        print(tokenizer.batch_decode(output, skip_special_tokens=True))
        total_stats["acceptance_rate"] += stats['ground_acceptance_count'] / \
            stats['decision_acceptance_count']
        total_stats["draft_generation_time"] += stats["draft_generation_time"]
        total_stats["token/second"] += output.shape[1] - input_ids.shape[1]
        total_stats["verification_time"] += stats["verification_time"]
        total_stats["decision_model_time"] += stats["decision_model_time"]
        total_stats["decision_acceptance_count"] += stats["decision_acceptance_count"]
        total_stats["ground_acceptance_count"] += stats["ground_acceptance_count"]
        total_stats["total_generation_round"] += stats["total_generation_round"]

    end = time.time()
    total_stats["token/second"] = total_stats["token/second"] / (end - start)
    total_stats["acceptance_rate"] = total_stats["acceptance_rate"] / len(prompts)
    print(f"total time spend is {end - start}")
    print(f"average decision accepted length is {total_stats['decision_acceptance_count'] / total_stats['total_generation_round']}")
    print(f"average ground accepted length is {total_stats['ground_acceptance_count'] / total_stats['total_generation_round']}")
    print('='*50)
    for k, v in total_stats.items():
        print(k, v)
    print()