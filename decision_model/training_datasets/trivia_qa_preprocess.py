from datasets import load_dataset
import json
dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

def transform_data(data):
    return {
        "question": data["question"],
    }

transformed_data = [transform_data(item) for item in dataset['test']]
with open("trivia_qa.json", "w") as f:
    json.dump(transformed_data, f, indent=2)