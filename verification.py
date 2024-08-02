import time
import torch
from model.llama_tree_attn import LlamaForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
from inference.fork_shape_tree_attn import *
from inference.strategies import *
from decision_model.decision_models import *
import matplotlib.pyplot as plt
import numpy as np

DRAFT_MODEL_NAME = 'JackFram/llama-68m'
# DRAFT_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DECISION_MODEL_NAME = "v1_soft"
DEBUG = False

if DECISION_MODEL_NAME == "v1_soft":
    DecisionModel = DecisionModelV1()
    DECISION_MODEL_PATH = "UCI-IASL-Transformer/decision_model/weights/decision_model_v1_soft.pt"

def plot_results(results,decision_model=DECISION_MODEL_NAME):
    thresholds = sorted(set(r['threshold'] for r in results))
    lengths = sorted(set(r['length'] for r in results))
    
    token_speeds = np.zeros((len(thresholds), len(lengths)))
    
    for r in results:
        i = thresholds.index(r['threshold'])
        j = lengths.index(r['length'])
        token_speeds[i, j] = r['token_per_second']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(token_speeds, cmap='viridis', aspect='auto')
    
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_xticklabels(lengths)
    ax.set_yticklabels(thresholds)
    
    plt.xlabel('Length')
    plt.ylabel('Threshold')
    plt.title('Token/s for Different Lengths and Thresholds')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Token/s', rotation=-90, va="bottom")
    
    for i in range(len(thresholds)):
        for j in range(len(lengths)):
            text = ax.text(j, i, f"{token_speeds[i, j]:.2f}",
                           ha="center", va="center", color="w")
    
    plt.tight_layout()
    plt.savefig(f'token_speed_heatmap_{decision_model}.png')
    plt.close()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)

# Load models
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

draft_model.eval()
target_model.eval()

# Parameters
max_new_tokens = 200
draft_model_temp = 0
target_model_temp = 0

width = 6
# Our accelerated strategy
strategy = NewTreeStrategy(
    draft_model=draft_model,
    target_model=target_model,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_new_tokens,
    greedy=False,
    using_decision_model=True,
    decision_model=DecisionModel,
    decision_model_path=DECISION_MODEL_PATH,
    decision_threshold= 0.4,
    config_depth= 7,
    config_width= width,
    draft_model_temp=draft_model_temp,
    target_model_temp=target_model_temp
)

def read_sentences_from_file():
    sentences = []
    with open('UCI-IASL-Transformer/sentences.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # Strip whitespace and skip empty lines
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)
    return sentences

prompts = [
    "The old oak tree stood tall, its branches reaching for the sky.",
    "She carefully placed the last piece of the puzzle, completing the image.",
    "The aroma of freshly baked bread filled the small bakery.",
    "The scientist peered into the microscope, searching for answers.",
    "Waves crashed against the rocky shore, sending spray into the air.",
    "The ancient manuscript revealed secrets long forgotten by time.",
    "Children laughed and played in the park on a sunny afternoon.",
    "The spaceship's engines roared to life as it prepared for launch.",
    "A lone wolf howled at the moon from atop a snow-covered hill.",
    "The artist dipped her brush in paint, ready to create a masterpiece.",
    "The detective examined the crime scene, looking for clues.",
    "Fireflies danced in the twilight, their lights twinkling like stars.",
    "The chef seasoned the dish with a pinch of exotic spices.",
    "A gentle breeze rustled the leaves of the cherry blossom trees.",
    "The marathon runner crossed the finish line, exhausted but triumphant.",
    "The magician waved his wand, making the rabbit disappear from the hat.",
    "Students hurried across the campus, late for their morning classes.",
    "The violinist closed her eyes, losing herself in the music.",
    "A shooting star streaked across the night sky, granting silent wishes.",
    "The archaeologist carefully brushed dirt from the ancient artifact.",
    "Raindrops pattered against the window pane, creating a soothing rhythm.",
    "The cat stretched lazily in a patch of warm sunlight on the carpet.",
    "The hiker reached the mountain peak, surveying the breathtaking view.",
    "A flock of geese flew overhead in a perfect V formation.",
    "The librarian stamped the due date on the last book of the day.",
    "The blacksmith hammered the red-hot metal, shaping it with skill.",
    "Colorful hot air balloons dotted the sky during the festival.",
    "The scuba diver explored the vibrant coral reef teeming with life.",
    "A rainbow appeared after the storm, arching across the sky.",
    "The clockmaker carefully adjusted the gears of the antique timepiece."
]
results = []

# Inside your nested loops, after calculating total_stats

# prompts = read_sentences_from_file()
threshold_to_experiment = [0.4,0.5,0.6,0.7,0.8]
# threshold_to_experiment = [0]
length_to_experiment = [3,4,5,6,7,8,9,10,11,12,13]
threshold_to_experiment = [0.4]
length_to_experiment = [7]
best_threshold = threshold_to_experiment[0]
best_length = length_to_experiment[0]
largest_token_per_second = float('-inf')
with torch.no_grad():
    for threshold in threshold_to_experiment:
        for length in length_to_experiment:
            total_stats = {'token/second': 0.0,"verification_time": 0.0,
                        'acceptance_rate': 0.0, "draft_generation_time": 0.0,
                        "ground_acceptance_count": 0, "total_generation_round": 0,
                        "decision_model_time": 0.0, "decision_acceptance_count": 0}

            strategy.decision_threshold = threshold
            strategy.stats = {'token/second': 0.0,'ground_acceptance_count': 0,
                        "draft_generation_time":0.0, "verification_time":0.0, 
                        "total_generation_round":0, "decision_model_time":0.0, 
                        "decision_acceptance_count":0}
            strategy.max_config = strategy.generate_fork_config(width = width,depth=length)
            strategy.tree_attn_self_mask = get_tree_attn_self_mask(strategy.max_config).to(device=strategy.draft_model_device) # type: ignore
            strategy.max_draft_len = len(strategy.max_config)-1
            prod_size = torch.cumprod(torch.tensor(strategy.max_config, dtype=torch.int), dim=0)
            prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
            strategy.prod_size = prod_size
            strategy.cumulative_prod_size = torch.cumsum(
                torch.tensor(prod_size), dim=0
            ).tolist()
            
            start = time.time()
            for sample_idx in tqdm(range(len(prompts))):
                prompt_text = prompts[sample_idx]
                inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids
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
            total_stats["acceptance_rate"] = total_stats["acceptance_rate"] / len(prompts)
            results.append({
                'threshold': threshold,
                'length': length,
                'token_per_second': total_stats['token/second']
            })
            print(f"acceptance rate is {total_stats['acceptance_rate']}")
            print(f"decision Model is {strategy.decision_model_name}")
            print(f"current threshold is {threshold}, current length is {length}")
            print(f"current best threshold: {best_threshold}, best length: {best_length}")
            print(f"total time spend is {end - start}")
            print(f"average decision accepted length is {total_stats['decision_acceptance_count'] / total_stats['total_generation_round']}")
            print(f"average ground accepted length is {total_stats['ground_acceptance_count'] / total_stats['total_generation_round']}")
            print(f"⭐token per second: {total_stats['token/second']}")
            print(f"verification time: {total_stats['verification_time']}")
            print(f"draft generation time: {total_stats['draft_generation_time']}")
            print(f"decision model time: {total_stats['decision_model_time']}\n")
            if total_stats["token/second"] > largest_token_per_second:
                best_threshold = threshold
                best_length = length 
                largest_token_per_second = total_stats["token/second"]
    print(f"with decision model {strategy.decision_model_name}\ntoken/sec: {largest_token_per_second }\nbest threshold: {best_threshold}, best length is {best_length}")
    plot_results(results,decision_model=strategy.decision_model_name)
for i,path in enumerate(strategy.max_token_path):
    print(f"path {i} the average longest accpeted length is {sum(path)/len(path)}")
# plt.figure(figsize=(10, 6))
# plt.hist(strategy.max_token_path, bins=width, edgecolor='black')

# # Add labels and title
# plt.xlabel('path number with maximum accepted tokens')
# plt.ylabel('Frequency')
# plt.title('target model initialized multi-candidate')

# # Add grid lines
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig(f"target_first_maximum_path_dist.png")

# # Show the plot
# plt.show()
