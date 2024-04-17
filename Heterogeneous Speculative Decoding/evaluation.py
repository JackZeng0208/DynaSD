from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from collections import Counter
import re
import string
from edge_speculative_decoding import edge_speculative_sampling

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
    ans = white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()
    return ans

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
    # print(f'prediction: {normalize_answer(prediction)}')
    # print(f'ground truth: {normalize_answer(ground_truth)}')
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, approx_model, SERVER_IP,client_id):
    f1 = exact_match = total = 0
    approx_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
    
    for example in tqdm(dataset):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        input_ids = approx_tokenizer.encode(input_str, return_tensors='pt')
        
        output = edge_speculative_sampling(
            prefix=input_ids,
            approx_model=approx_model,
            SERVER_IP=SERVER_IP,
            max_len=50,
            gamma=4,
            client_id=client_id
        )
        
        pred_answer = approx_tokenizer.batch_decode(output)[0].split("Answer:")[1].strip()
        ground_truths = example["answer"]["aliases"]
        exact_match += metric_max_over_ground_truths(exact_match_score, pred_answer, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, pred_answer, ground_truths)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

if __name__ == "__main__":
    SERVER_IP = '192.168.0.132'
    client_id = 'client_2'
    approx_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="auto", trust_remote_code=True)
    approx_model.to('cuda:0')
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    dataset = dataset['validation'].select(range(1000))
    eval_result = evaluate(dataset, approx_model, SERVER_IP,client_id=client_id)
    with open("eval_result_speculative_decoding_triviaQA.txt", 'w') as f:
        f.write(f"Test results: {eval_result}\n")