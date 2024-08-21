from human_eval.data import read_problems
import time
import torch
from model.llama_tree_attn import LlamaForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
from inference.fork_shape_tree_attn import *
from inference.strategies import *
from DynaSD.decision_models import *
import matplotlib.pyplot as plt
import numpy as np

SOFT_LABEL = True
# DRAFT_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
DRAFT_MODEL_NAME = 'JackFram/llama-68m'
TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# TARGET_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

if DRAFT_MODEL_NAME == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and TARGET_MODEL_NAME == "lmsys/vicuna-7b-v1.5":
    DECISION_MODEL_NAME = "tinyllama_vicuna_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1_Tinyllama().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'JackFram/llama-68m' and TARGET_MODEL_NAME == "lmsys/vicuna-7b-v1.5":
    DECISION_MODEL_NAME = "llama_68m_vicuna_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'JackFram/llama-68m' and TARGET_MODEL_NAME == "meta-llama/Llama-2-7b-chat-hf":
    DECISION_MODEL_NAME = "llama_68m_llama2_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if DRAFT_MODEL_NAME == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' and TARGET_MODEL_NAME == "meta-llama/Llama-2-7b-chat-hf":
    DECISION_MODEL_NAME = "tinyllama_llama2_7b"
    if SOFT_LABEL == True:
        decision_model = DecisionModelV1_Tinyllama().cuda()
    else:
        decision_model = DecisionModelVTopk().cuda()

if SOFT_LABEL == True:
    file_name = f"soft_{DECISION_MODEL_NAME}"
else:
    file_name = f"hard_{DECISION_MODEL_NAME}"

draft_model = LlamaForCausalLM.from_pretrained(
    DRAFT_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=0,
)

target_model = LlamaForCausalLM.from_pretrained(
    TARGET_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

max_new_tokens = 200
DECISION_MODEL_PATH = f'/home/iasl-transformers/DynaSD/decision_model/{file_name}.pt'
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
strategy = NewTreeStrategy(
    draft_model=draft_model,
    target_model=target_model,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_new_tokens,
    greedy=False,
    using_decision_model= True,
    decision_model=decision_model,
    decision_model_path=DECISION_MODEL_PATH,
    decision_threshold= 0.5,
    config_depth= 10,
    config_width= 6,
    draft_model_temp=0,
    target_model_temp=0,
    soft_label=SOFT_LABEL
)


problems = read_problems()
total_stats = {'token/second': 0.0,"verification_time": 0.0,
            'acceptance_rate': 0.0, "draft_generation_time": 0.0,
            "ground_acceptance_count": 0, "total_generation_round": 0,
            "decision_model_time": 0.0, "decision_acceptance_count": 0}
start = time.time()
for task_id in tqdm(problems):
    input_ids = problems[task_id]['prompt']
    input_ids = tokenizer(input_ids, return_tensors='pt').to('cuda')
    input_ids = input_ids.input_ids
    output, stats = strategy.generation_loop(input_ids=input_ids)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
    total_stats["acceptance_rate"] += stats['ground_acceptance_count'] / \
        stats['decision_acceptance_count']
    total_stats["draft_generation_time"] = stats["draft_generation_time"]
    total_stats["token/second"] += output.shape[1] - input_ids.shape[1]
    total_stats["verification_time"] = stats["verification_time"]
    total_stats["ground_acceptance_count"] += stats["ground_acceptance_count"]
    total_stats["decision_model_time"] = stats["decision_model_time"]
    total_stats["decision_acceptance_count"] += stats["decision_acceptance_count"]
    total_stats["total_generation_round"] = stats["total_generation_round"]
end = time.time()
total_stats["token/second"] = total_stats["token/second"] / (end - start)
total_stats["acceptance_rate"] = total_stats["acceptance_rate"] / len(problems)
print(f"current draft model{DRAFT_MODEL_NAME}, current target model is {TARGET_MODEL_NAME}")
print(f"current decision model is {DECISION_MODEL_PATH}")
print(f"acceptance rate is {total_stats['acceptance_rate']}")
print(f"total time spend is {end - start}")
print(f"average decision accepted length is {total_stats['decision_acceptance_count'] / total_stats['total_generation_round']}")
print(f"average ground accepted length is {total_stats['ground_acceptance_count'] / total_stats['total_generation_round']}")
print(f"⭐token per second: {total_stats['token/second']}")
print(f"verification time: {total_stats['verification_time']}")
print(f"draft generation time: {total_stats['draft_generation_time']}")
print(f"decision model time: {total_stats['decision_model_time']}\n")



