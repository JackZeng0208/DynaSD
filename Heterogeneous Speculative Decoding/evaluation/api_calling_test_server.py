import zmq
# from collections import Counter
# import re
# import string
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from datasets import load_dataset
import pynvml
import threading
import csv
pynvml.nvmlInit()

# Copy from TriviaQA Evaluation Code
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
# def normalize_answer(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
#     def white_space_fix(text):
#         return ' '.join(text.split())
#     def handle_punc(text):
#         exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
#         return ''.join(ch if ch not in exclude else ' ' for ch in text)
#     def lower(text):
#         return text.lower()
#     def replace_underscore(text):
#         return text.replace('_', ' ')
#     return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

# # Copy from TriviaQA Evaluation Code
# # https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

# def exact_match_score(prediction, ground_truth):
#     return normalize_answer(prediction) == normalize_answer(ground_truth)

# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)

def evaluate(dataset, model, tokenizer, socket: zmq.Socket):
    device = torch.device("cuda:0")
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    with open(f"gpu_utilization_api_calling.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        while True:
            gpu_utilization = []
            def capture_gpu_utilization(stop_event):
                # Adjust the sample interval as needed (in seconds) -> 1ms
                sample_interval = 0.1
                while not stop_event.is_set():
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    gpu_utilization.append(utilization)
                    time.sleep(sample_interval)
            # Start capturing GPU utilization in a separate thread
            stop_event = threading.Event()
            gpu_thread = threading.Thread(target=capture_gpu_utilization, args=(stop_event,))
            gpu_thread.start()
            
            question = socket.recv_pyobj()
            start_time = time.time()
            # print(f"Received question: {question}")
            # Find the corresponding example in the dataset
            example = next((ex for ex in dataset if ex["question"] == question), None)
            if example is None:
                print(f"Question not found in the dataset: {question}")
                result = {
                    "output": "None",
                    "inference_time": 0,
                    "gpu_utilization": []
                }
                socket.send_json(result)
                continue
            # ground_truths = example["answer"]["aliases"]
            input_str = f"Question: {question}\nAnswer:"
            input_ids = tokenizer.encode(input_str, return_tensors="pt").to("cuda:0")
            # print(input_ids.shape)
            output = model.generate(input_ids, max_new_tokens=128)
            # print(f"GPU utilization: {gpu_utilization}")
            pred_answer = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(f"Generated answer: {pred_answer}")
            # print(f"Inference time: {inference_time} seconds")
            # em_score = metric_max_over_ground_truths(exact_match_score, pred_answer, ground_truths)
            # f1 = metric_max_over_ground_truths(f1_score, pred_answer, ground_truths)
            end_time = time.time()
            inference_time = end_time - start_time
            result = {
                "output": pred_answer,
                "inference_time": inference_time,
            }
            socket.send_pyobj(result)
            # Stop capturing GPU utilization
            stop_event.set()
            gpu_thread.join()
            writer.writerow([gpu_utilization])

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
dataset = dataset['validation'].select(range(5000))

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:1919")
print("Server is running...")

evaluate(dataset, model, tokenizer, socket)
