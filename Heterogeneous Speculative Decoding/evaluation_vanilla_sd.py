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

def evaluate(dataset, model_name, server_ip, port, client_id):
    f1 = exact_match = total = 0
    approx_model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)
    approx_tokenizer = AutoTokenizer.from_pretrained(
        model_name)
    approx_model.to('cuda:0')
    client = HeteroSpeculativeDecoding()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # Set send buffer size to 1 MB
    socket.setsockopt(zmq.SNDBUF, 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
    socket.connect(f"tcp://{server_ip}:{port}")
    total_acceptace_rate = 0
    total_token_speed = 0
    for example in tqdm(dataset):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        input_ids = approx_tokenizer.encode(input_str, return_tensors='pt')

        output, acc_rate, token_speed = client.edge_speculative_decoding(
            input_ids=input_ids,
            draft_model=approx_model,
            edge_socket=socket,
            max_len=128,
            client_id=client_id
        )
        # pred_answer = approx_tokenizer.batch_decode(
        #     output)[0].split("Answer:")[1].strip()
        # ground_truths = example["answer"]["aliases"]
        # exact_match += metric_max_over_ground_truths(
        #     exact_match_score, pred_answer, ground_truths)
        # f1 += metric_max_over_ground_truths(f1_score,
        #                                     pred_answer, ground_truths)
        total += 1
        total_acceptace_rate += acc_rate
        total_token_speed += token_speed
    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total
    socket.send_pyobj({"end": True})
    socket.recv_pyobj()
    socket.close()
    return total_acceptace_rate / total, total_token_speed / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Evaluation")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Draft model name")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa",
                        help="Huggingface dataset name (ex: mandarjoshi/trivia_qa)")
    parser.add_argument("--range", nargs=2, type=int,
                        default=[666, 1332], help="Range of dataset to evaluate")
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

    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    dataset = dataset['validation']
    dataset = dataset.filter(lambda example: len(example["question"]) <= 128)
    
    dataset = dataset['validation'].select([i for i in range(args.range[0], args.range[1])])
    acc_rate, speed = evaluate(dataset, args.model_name,
                           args.server_ip, args.port, args.client_id)
    with open(f"vanilla_sd_benchmark_{os.getlogin()}_triviaQA.txt", 'w') as f:
        f.write(f"Acceptance Rate: {acc_rate}, Token Generation Speed: {speed}")
