import zmq
from tqdm import tqdm
from collections import Counter
import re
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from datasets import load_dataset

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:1919")
print("Server is running...")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
dataset = dataset['validation'].select(range(1000))

# Copy from TriviaQA Evaluation Code
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

# Copy from TriviaQA Evaluation Code
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    # print(f"Ground Truth = {ground_truths}")
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

while True:
    question = socket.recv_string()
    print(f"Received question: {question}")

    # Find the corresponding example in the dataset
    example = next((ex for ex in dataset if ex["question"] == question), None)
    if example is None:
        print(f"Question not found in the dataset: {question}")
        result = {
            "inference_time": 0,
            "exact_match": 0,
            "f1": 0
        }
        socket.send_json(result)
        continue

    ground_truths = example["answer"]["aliases"]

    input_str = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to("cuda:0")

    start_time = time.time()
    output = model.generate(input_ids, max_length=50)
    end_time = time.time()

    pred_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated answer: {pred_answer}")

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time} seconds")

    em_score = metric_max_over_ground_truths(exact_match_score, pred_answer, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, pred_answer, ground_truths)

    result = {
        "inference_time": inference_time,
        "exact_match": em_score,
        "f1": f1
    }

    socket.send_json(result)