from datasets import load_dataset
from transformers import AutoTokenizer
from model.llama_tree_attn import LlamaForCausalLM
from tqdm import tqdm
from sd import SpeculativeDecoding
import argparse
import os
import torch

def evaluate(dataset, draft_model_name, target_model_name, max_len, gamma, top_k, top_p):
    total = 0
    draft_model = LlamaForCausalLM.from_pretrained(
        draft_model_name, torch_dtype=torch.float16, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        draft_model_name, trust_remote_code=True)
    draft_model.to('cuda:0')

    target_model = LlamaForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.float16, trust_remote_code=True)
    target_model.to('cuda:0')

    speculative_decoding = SpeculativeDecoding()

    total_acceptance_rate = 0
    total_token_speed = 0
    for example in tqdm(dataset):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        input_ids = draft_tokenizer.encode(input_str, return_tensors='pt')
        output, acceptance_rate, token_speed = speculative_decoding.speculative_decoding(
            input_ids=input_ids,
            draft_model=draft_model,
            target_model=target_model,
            max_len=max_len,
            gamma=gamma,
            top_k=top_k,
            top_p=top_p
        )
        total += 1
        total_acceptance_rate += acceptance_rate
        total_token_speed += token_speed

    return total_acceptance_rate / total, total_token_speed / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Speculative Decoding Evaluation")
    parser.add_argument("--draft_model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Draft model name")
    parser.add_argument("--target_model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Target model name")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa",
                        help="Huggingface dataset name (ex: mandarjoshi/trivia_qa)")
    parser.add_argument("--range", nargs=2, type=int, default=[0, 700],
                        help="Range of dataset to evaluate")
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
    dataset = dataset.select([i for i in range(args.range[0], args.range[1])])

    acc_rate, speed = evaluate(dataset, args.draft_model_name, args.target_model_name,
                               args.max_len, args.gamma, args.top_k, args.top_p)

    with open(f"speculative_decoding_benchmark_{os.getlogin()}_triviaQA.txt", 'w') as f:
        f.write(f"Acceptance Rate: {acc_rate}, Token Generation Speed: {speed}")