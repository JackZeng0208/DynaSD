from medusa.model.medusa_model import MedusaModel
from transformers import AutoTokenizer
import torch
import time
from tqdm import tqdm
MODEL_NAME = "FasterDecoding/medusa-vicuna-7b-v1.3"
model = MedusaModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
tokenizer = model.get_tokenizer()
# Parameters
max_length = 200
temperature = 0.1
model.eval()
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

def generate_responses(prompts, max_length=max_length, temperature=temperature):
    total_inference_time = 0
    total_token_length = 0
    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            start_time = time.time()
            output = model.medusa_generate(input_ids, max_steps=max_length, temperature=temperature)
            end_time = time.time()
        total_inference_time += end_time - start_time
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        total_token_length += output.shape[1] - input_ids.shape[1]
        print(response)
    tokens_per_second = total_token_length / total_inference_time
    return total_inference_time, tokens_per_second
inference_time, speed = generate_responses(prompts)
print(f"Total inference time: {inference_time}")
print(f"Tokens/s: {speed}")