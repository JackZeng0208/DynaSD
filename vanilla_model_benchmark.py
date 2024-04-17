from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from tqdm import tqdm
from collections import Counter
import re
import string
from vllm import LLM

# generation_config = GenerationConfig(
#     max_length = 100
#     )

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

def evaluate(model, dataset):
    f1 = exact_match = total = 0
    for example in tqdm(dataset):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        pred_answer = model.generate(input_str)
        # print(pred_answer)
        pred_answer = pred_answer[0].outputs[0].text.strip()
        ground_truths = example["answer"]["aliases"]
        exact_match += metric_max_over_ground_truths(exact_match_score, pred_answer, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, pred_answer, ground_truths)
        total += 1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

# Load the TriviaQA dataset
dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
dataset = dataset['validation']

model = LLM(
    "meta-llama/Llama-2-7b-chat-hf",
    max_context_len_to_capture=256,
    tensor_parallel_size=2,
    tokenizer="meta-llama/Llama-2-7b-chat-hf"
)

eval_result = evaluate(model, dataset)

with open("eval_result_vllm_llama_triviaQA.txt", 'w') as f:
    f.write(f"Test results: {eval_result}\n")