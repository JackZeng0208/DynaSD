import json
def extract_questions_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        questions = []
        for entry in data:
            if 'question' in entry:
                questions.append(entry['question'])
            if 'goal' in entry:
                questions.append(entry['goal'])
    return questions


file1 = 'UCI-IASL-Transformer/evaluation/alpaca_clean.json'
file2 = 'UCI-IASL-Transformer/evaluation/piqa.json'
file3 = 'UCI-IASL-Transformer/evaluation/trivia_qa.json'


output_file = 'combined_training_questions.json'


questions_file1 = extract_questions_from_file(file1)
questions_file2 = extract_questions_from_file(file2)
questions_file3 = extract_questions_from_file(file3)

all_questions = questions_file1 + questions_file2 + questions_file3

with open(output_file, 'w') as file:
    json.dump(all_questions, file, indent=4)