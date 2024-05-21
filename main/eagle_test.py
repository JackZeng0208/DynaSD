import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from .utils import prepare_logits_processor, generate_tree_buffers, initialize_tree
from .ea_model import EaModel

@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None
    tree_config: Optional[Tuple] = None

@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    sequences: torch.LongTensor = None
    draft_model_accept_indices: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None

def get_tree_attn_self_mask(k_config: Tuple[int]):
    k_config = torch.tensor(k_config, dtype=torch.int)
    prod_size = torch.cumprod(k_config, dim=0)
    mask_size = prod_size.sum().item()
    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    attn_mask = attn_mask.diagonal_scatter(torch.ones(mask_size))
    idx_queue = [(0, None, idx) for idx in range(k_config[0])]
    while idx_queue:
        depth, parent, idx = idx_queue.pop(0)
        if parent is not None:
            attn_mask[idx, :parent + 1] = attn_mask[parent, :parent + 1]
        if depth < len(k_config) - 1:
            idx_base = prod_size[:depth].sum().item()
            child_idx_base = prod_size[:depth + 1].sum().item()
            for child_idx_bias in range(k_config[depth + 1]):
                real_child_idx = (idx - idx_base) * k_config[depth + 1] + child_idx_base + child_idx_bias
                idx_queue.append((depth + 1, idx, real_child_idx))
    return attn_mask

def initialize_past_key_values(model):
    past_key_values = [None] * model.config.num_hidden_layers
    past_key_values_data = [None] * model.config.num_hidden_layers
    current_length_data = torch.zeros(1, dtype=torch.long, device=model.device)
    return past_key_values, past_key_values_data, current_length_data

class EAGLEDraftModel(torch.nn.Module):
    def __init__(self, draft_model, tree_config):
        super().__init__()
        self.draft_model = draft_model
        self.tree_config = tree_config
        self.tree_attn_self_mask = get_tree_attn_self_mask(tree_config)
        
    def forward(self, input_ids, past_key_values=None):
        outputs = self.draft_model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        return next_tokens, outputs.past_key_values

    def generate(self, input_ids, past_key_values=None):
        sequences = input_ids
        for _ in range(len(self.tree_config)):
            next_tokens, past_key_values = self(input_ids, past_key_values)
            input_ids = torch.cat((input_ids, next_tokens), dim=1)
            sequences = torch.cat((sequences, next_tokens), dim=1)
        return sequences, past_key_values

def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor=None):
    # Implement candidate generation logic
    candidates = tree_indices[sample_token]
    cart_candidates_prob = F.softmax(tree_logits[sample_token], dim=-1)
    tree_candidates = retrieve_indices[candidates]
    return candidates, cart_candidates_prob, tree_candidates

def tree_decoding(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices_head):
    # Implement logic to get logits
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    hidden_state_new = outputs.hidden_states[-1]
    return logits, hidden_state_new, outputs

def evaluate_posterior(logits, candidates, logits_processor, cart_candidates_prob, tree_logits, p_indices, tree_candidates, b_indices):
    # Implement logic to find the best candidate
    best_candidate = torch.argmax(cart_candidates_prob)
    accept_length = (logits[candidates[best_candidate]] > 0).sum()
    sample_p = torch.softmax(logits, dim=-1)
    return best_candidate, accept_length, sample_p

def update_inference_inputs(input_ids, candidates, best_candidate, accept_length, retrieve_indices, logits_processor, logits, tree_logits, new_token, past_key_values_data, current_length_data, model, hidden_state, hidden_state_new, sample_p):
    # Implement logic to update input IDs
    input_ids = torch.cat([input_ids, candidates[best_candidate].unsqueeze(0)], dim=1)
    tree_logits = logits
    new_token += 1
    hidden_state = hidden_state_new
    sample_token = candidates[best_candidate]
    return input_ids, tree_logits, new_token, hidden_state, sample_token

class EAGLESpeculativeDecoding:
    def __init__(self, model, tokenizer, tree_choices, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.tree_choices = tree_choices
        self.temperature = temperature
        self.time_spend_on_draft_model_generation = 0
        self.time_spend_on_target_model_forward = 0

    @torch.no_grad()
    def sampling_without_kvcache(self, draft_tokens):
        target_model_history = self.model(draft_tokens).logits
        for i in range(target_model_history.shape[-2]):
            target_model_history[:, i, :] = F.log_softmax(target_model_history[:, i, :] / self.temperature, dim=-1)
        return target_model_history

    def speculative_decoding(self, input_ids, max_len, gamma=4):
        device = torch.device("cuda:0")
        self.model.to(device)
        seq_len = input_ids.shape[1]
        T = seq_len + max_len

        input_ids = input_ids.to(device)
        start_time = time.time()

        total_draft_generate_count = 0
        while input_ids.shape[1] < T:
            prefix_len = input_ids.shape[1]

            draft_generate_start_time = time.time()
            tree_buffers = generate_tree_buffers(self.tree_choices, device)
            tree_logits, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self.model, tree_buffers["tree_attn_mask"], None, None
            )

            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                None
            )
            logits, hidden_state_new, outputs = tree_decoding(
                self.model,
                tree_candidates,
                None,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, None, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                None,
                logits,
                tree_logits,
                0,
                None,
                None,
                self.model,
                hidden_state,
                hidden_state_new,
                sample_p
            )
            draft_generate_end_time = time.time()
            self.time_spend_on_draft_model_generation += draft_generate_end_time - draft_generate_start_time
            total_draft_generate_count += gamma

            target_forward_time = time.time()
            target_model_history_tensor = self.sampling_without_kvcache(input_ids)
            finish_target_forward_time = time.time()
            self.time_spend_on_target_model_forward += finish_target_forward_time - target_forward_time

            if new_token > 1024 or input_ids.shape[1] > 1960:
                break

        end_time = time.time()
        token_generate_speed = (input_ids.shape[-1] - seq_len) / (end_time - start_time)
        acceptance_rate = gamma / total_draft_generate_count

        print(f"Total time spent on speculative decoding: {end_time - start_time}")
        print(f"Token Generation Speed: {token_generate_speed} tokens/s")
        print(f"Acceptance Rate: {acceptance_rate}")
        torch.cuda.empty_cache()

        return input_ids, acceptance_rate, token_generate_speed

@torch.inference_mode()
def get_model_answers(model, tokenizer, question, temperature, tree_choices):
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    torch.cuda.synchronize()
    start_time = time.time()
    input_ids = tokenizer(question, return_tensors='pt').input_ids
    ea_decoding = EAGLESpeculativeDecoding(model, tokenizer, tree_choices, temperature)
    output_ids, acceptance_rate, token_generate_speed = ea_decoding.speculative_decoding(
        input_ids.cuda(), max_len=128, gamma=4
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output}")
    print(f"Total time: {total_time}")
    print(f"Tokens per second: {token_generate_speed}")
    return output

def evaluate(dataset, base_model_path, ea_model_path, tree_choice):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    for example in tqdm(dataset):
        question = example["question"]
        print(f"\nQuestion: {question}\n")
        get_model_answers(
            model=model,
            tokenizer=tokenizer,
            question=question,
            temperature=1,
            tree_choices=tree_choice,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Speculative Decoding Evaluation")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Base model path")
    parser.add_argument("--ea_model_path", type=str, default="yuhuili/EAGLE-llama2-chat-7B", help="EAGLE model path")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa", help="Huggingface dataset name")
    parser.add_argument("--range", nargs=2, type=int, default=[0, 1000], help="Range of dataset to evaluate")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, "rc.nocontext")['validation']
    dataset = dataset.filter(lambda example: len(example["question"]) <= 128)
    dataset = dataset.select([i for i in range(args.range[0], args.range[1])])

    mc_sim_7b_63 = [[0],[1],[2],[3],[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]
                ,[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,2,0],[0,2,1],[1,0,0],
                [0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,0,0],[0,0,0,0,1]]
    
    evaluate(dataset, args.base_model_path, args.ea_model_path, mc_sim_7b_63)
