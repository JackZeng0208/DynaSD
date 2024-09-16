from datasets import load_dataset
import json

def preprocess_alpaca_dataset(dataset):
    data = dataset['train']

    processed_data = []
    for item in data:
        instruction = item['instruction'].strip()
        input_text = item['input'].strip() if item['input'] else ''
        output_text = item['output'].strip() if item['output'] else ''
        combined_text = {"question": instruction, "input": input_text, "output": output_text}
        processed_data.append(combined_text)

    output_file = "alpaca_clean.json"
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=2)

dataset = load_dataset("yahma/alpaca-cleaned")
preprocess_alpaca_dataset(dataset)