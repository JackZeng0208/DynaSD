import time
import torch
from inference.generate import SpeculativeGenerator
from model.llama_tree_attn import LlamaForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
from inference.strategies import *
import matplotlib.pyplot as plt

DRAFT_MODEL_NAME = 'JackFram/llama-68m'
# DRAFT_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
# k_config = (5, 1, 1, 1, 1, 1)
# k_config = (4,2,2,1,1)
width = 6
k_config = (width,1,1,1,1)
max_new_tokens = 200

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

# Parameters
draft_model_temp = 0
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

# Metrics initialization
acceptance_count = 0
draft_token_count = 0
invocation_count = 0
total_tokens = 0

start_time = time.time()
with torch.no_grad():
    for prompt_text in tqdm(prompts):
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        output = generator.generate(input_ids)
        output_text = tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        print(output_text)
        total_tokens += output.sequences.shape[1] - input_ids.shape[1]
        acceptance_count += output.acceptance_count
        draft_token_count += output.draft_token_count
        invocation_count += output.invocation_count

end_time = time.time()
run_time = end_time - start_time
latency = run_time / (acceptance_count + invocation_count)
acceptance_rate = acceptance_count / draft_token_count
block_efficiency = 1 + acceptance_count / invocation_count
# print(f"latency {latency}")
# print(f"acceptance rate is {acceptance_rate}")
# print(f"total run time is {run_time}")
for i,path in enumerate(generator.stats):
    print(f"path {i} the average longest accepted length is {sum(path)/len(path)}")
# plt.figure(figsize=(10, 6))
# plt.hist(generator.stats, bins=width, edgecolor='black')

# # Add labels and title
# plt.xlabel('path number with maximum accepted tokens')
# plt.ylabel('Frequency')
# plt.title('draft model initialized multi-candidate')

# # Add grid lines
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig(f"draft_first_maximum_path_dist.png")
# # Show the plot
# plt.show()

print("\nResults for MCSD:")
print(f"Run Time: {run_time:.2f}s")
print(f"Latency: {latency*1000:.2f}ms")
print(f"Acceptance Rate: {acceptance_rate:.2f}")
print(f"Tokens/s: {total_tokens / run_time:.2f}")
print()
