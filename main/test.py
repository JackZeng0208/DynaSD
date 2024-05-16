from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
# from sd import SpeculativeDecoding, TreeAttentionSpeculativeDecoding
import argparse
import time
import torch

def evaluate(dataset, draft_model_name, target_model_name, max_len, gamma, top_k, top_p):
    total = 0
    draft_model = LlamaForCausalLM.from_pretrained(draft_model_name, torch_dtype="auto").to("cuda:0")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = LlamaForCausalLM.from_pretrained(target_model_name, torch_dtype="auto").to("cuda:0")
    for example in tqdm(dataset):
        question = example["question"]
        start_time = time.time()
        input_str = f"Question: {question}\nAnswer:"
        input_ids = target_tokenizer.encode(input_str, return_tensors='pt').to("cuda:0")
        outputs = target_model.generate(input_ids, 
                                        assistant_model=draft_model, 
                                        do_sample=True, 
                                        temperature=1,
                                        max_new_tokens=max_len,
                                        top_k=top_k,
                                        top_p=top_p,)
        # ans = target_tokenizer.decode(outputs, skip_special_tokens=True)
        end_time = time.time()
        print(f"Token Generation Speed: { (len(outputs[0]) - len(input_ids[0]))  / (end_time - start_time)}")
        total += 1
        
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
    evaluate(dataset, args.draft_model_name, args.target_model_name,
                               args.max_len, args.gamma, args.top_k, args.top_p)