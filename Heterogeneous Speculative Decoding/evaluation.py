from datasets import load_dataset
from transformers import AutoTokenizer
from model.llama_tree_attn import LlamaForCausalLM
from tqdm import tqdm
from collections import Counter
import re
import string
from hetero_speculative_decoding import HeteroSpeculativeDecoding
import argparse
import os
import torch
import zmq


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
    ans = white_space_fix(remove_articles(
        handle_punc(lower(replace_underscore(s))))).strip()
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


def evaluate(dataset, model_name, server_ip, port, client_id):
    f1 = exact_match = total = 0
    approx_model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True)
    approx_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    approx_model.to('cuda:0')
    client = HeteroSpeculativeDecoding()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # Set send buffer size to 1 MB
    socket.setsockopt(zmq.SNDBUF, 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
    socket.connect(f"tcp://{server_ip}:{port}")
    total_acceptace_rate = 0
    for example in tqdm(dataset):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        input_ids = approx_tokenizer.encode(input_str, return_tensors='pt')

        output, acceptance_rate = client.edge_tree_attn_speculative_decoding(
            input_ids=input_ids,
            draft_model=approx_model,
            edge_socket=socket,
            max_len=128,
            client_id=client_id
        )
        total_acceptace_rate += acceptance_rate
        pred_answer = approx_tokenizer.batch_decode(
            output)[0].split("Answer:")[1].strip()
        ground_truths = example["answer"]["aliases"]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, pred_answer, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            pred_answer, ground_truths)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print(f"Total acceptance rate: {total_acceptace_rate / total}")
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Evaluation")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Draft model name")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa",
                        help="Huggingface dataset name (ex: mandarjoshi/trivia_qa)")
    parser.add_argument("--range", nargs=2, type=int,
                        default=[0, 1000], help="Range of dataset to evaluate")
    parser.add_argument("--server_ip", type=str, default="192.168.0.132")
    parser.add_argument("--port", type=str, default="1919",
                        help="Server port number")
    parser.add_argument("--client_id", type=str,
                        default=os.getlogin(), help="Client ID")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--input_text", type=str,
                        default="Please write an introduction about UC Irvine:")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, "rc.nocontext")
    dataset = dataset['validation'].select([i for i in range(args.range[0], args.range[1])])
    eval_result = evaluate(dataset, args.model_name,
                           args.server_ip, args.port, args.client_id)
    # with open("eval_result_speculative_decoding_triviaQA.txt", 'w') as f:
    #     f.write(f"Test results: {eval_result}\n")
