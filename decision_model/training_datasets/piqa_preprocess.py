from datasets import load_dataset
import json

def transform_data(data):
    correct_answer = data["sol1"] if data["label"] == 0 else data["sol2"]
    return {
        "goal": data["goal"],
        "correct_answer": correct_answer
    }

dataset = load_dataset("ybisk/piqa")
dataset = dataset["train"]
processed_data = [transform_data(data) for data in dataset]
with open("piqa.json", "w") as file:
    json.dump(processed_data, file, indent=2)
print(f"Number of examples: {len(processed_data)}")