from datasets import load_dataset
import json
import random

def preprocess_alpaca_dataset(dataset, num_samples=5000, max_input_length=128):
    data = dataset['train']

    processed_data = []
    for item in data:
        instruction = item['instruction'].strip()
        input_text = item['input'].strip() if item['input'] else ''
        combined_text = ""
        if input_text == '':
            combined_text = f"Instruction: {instruction}\nAnswer:"
        else:
            combined_text = f"Instruction: {instruction}\nInput: {input_text}\nAnswer:".strip()
        
        # Ignore long input questions
        if len(input_text.split()) <= max_input_length:
            processed_data.append(combined_text)

    selected_data = random.sample(processed_data, num_samples)
    output_file = "alpaca_clean.json"
    with open(output_file, 'w') as file:
        json.dump(selected_data, file, indent=2)
    print(f"Preprocessed and selected {num_samples} samples. Saved to {output_file}.")

dataset = load_dataset("yahma/alpaca-cleaned")
preprocess_alpaca_dataset(dataset)