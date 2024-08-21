from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import time
from tqdm import tqdm
from inference.sd_single_candidate import greedy_speculative_sampling

# Load model and tokenizer
from human_eval.data import read_problems
DRAFT_MODEL_NAME = 'JackFram/llama-68m'
# DRAFT_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
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

using_speculative_decoding = True

draft_model.eval()
target_model.eval()

# Parameters
max_length = 200
temperature = 1

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

problems = read_problems()
def generate_responses(prompts, max_length=max_length, temperature=temperature):
    total_inference_time = 0
    total_token_length = 0
    for idx in tqdm(problems):
        prompt_text = problems[idx]['prompt']
        
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to('cuda')
        if not using_speculative_decoding:
            with torch.no_grad():
                start_time = time.time()
                output = target_model.generate(input_ids, max_new_tokens =max_length, temperature=temperature, num_return_sequences=1)
                end_time = time.time()
        else:
            with torch.no_grad():
                start_time = time.time()
                # output = target_model.generate(input_ids, max_new_tokens =max_length, temperature=temperature, num_return_sequences=1, assistant_model=draft_model)
                output,current_acceptance_rate = greedy_speculative_sampling(input_ids,draft_model,target_model,max_len=200)
                print(f"current acceptance rate is {current_acceptance_rate}")
                end_time = time.time()
        total_inference_time += end_time - start_time
        response = tokenizer.batch_decode(output, skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        print(response)
    tokens_per_second = total_token_length / total_inference_time
    return total_inference_time, tokens_per_second
inference_time, speed = generate_responses(prompts)
print(f"Total inference time: {inference_time}")
print(f"Tokens/s: {speed}")