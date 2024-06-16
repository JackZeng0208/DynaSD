# Code based on https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy
from kv_cache import KVCacheModel
from spec_utils import norm_logits, sample, max_fn, collect_training_data
# from globals import Decoder
import time
# tokenizer = AutoTokenizer()
@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None, collect_data : bool = False, file_path : str = None):
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.
        collect_data (bool, optional): Collecting training dataset for decision model, defaults to False.
        file_path (str, optional): The file path to save the collected data, defaults to None.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    draft_forward_count = 0
    draft_generation_time = 0
    target_foward_time = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        draft_start = time.time()
        # print(f"shape of prefix is {prefix.shape}")
        x = approx_model_cache.generate(prefix, gamma)
        draft_end = time.time()
        draft_generation_time += draft_end - draft_start
        
        target_start = time.time()
        _ = target_model_cache.generate(x, 1)
        target_end = time.time()
        target_foward_time += target_end - target_start
        
        draft_forward_count += gamma
        
        n = prefix_len + gamma - 1
        
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]

            # TODO: Calculate logit difference
            # TODO (from Yixiao Zeng): not sure about
            logits = approx_model_cache._prob_history[:, prefix_len + i - 1, :]
            top2_logits = torch.topk(logits, 2, dim=-1).values
             # Difference between highest and second highest logit
            logit_difference = (top2_logits[:, 0] - top2_logits[:, 1]).item()

            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # print(f"shape of approx_model_cache {approx_model_cache._prob_history[:, prefix_len + i - 1, :].shape}")
                if collect_data:
                    collect_training_data(approx_model_cache._prob_history[:, prefix_len + i - 1, :],j,False,(approx_model_cache._prob_history[:, prefix_len + i - 1, j]), file_path)
                # reject
                n = prefix_len + i - 1
                break
            
            # if verbose:
            #     # print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            if collect_data:
                collect_training_data(approx_model_cache._prob_history[:, prefix_len + i - 1, :],j,True,(approx_model_cache._prob_history[:, prefix_len + i - 1, j]), file_path)

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            # if verbose:
            #     print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            # if verbose:
            #     print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)

    
    # print(f"acceptance rate is {accepted_count/(prefix.shape[-1] - seq_len)} generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    stats = {'acceptance_rate':accepted_count/draft_forward_count,
             'target_forward_count':target_sample_count+resample_count,
             'draft_forward_count':draft_forward_count,
             'draft_generation_time':draft_generation_time,
             'target_foward_time':target_foward_time}
    # stats = None
    return prefix,stats


def target_first_speculative_decoding(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None):
   
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    draft_forward_count = 0
    draft_generation_time = 0
    target_foward_time = 0
    
    target_start = time.time()
    target_dist = target_model_cache._forward_with_kvcache(prefix, False)
    target_end = time.time()
    target_foward_time += target_end - target_start
    target_sample_count += 1

    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        #FIXME: may have error by just add one here
        prefix_len = prefix.shape[1] +1
        draft_start = time.time()
        x = beam_draft_generation(gamma,prefix,target_dist,approx_model_cache)
        draft_end = time.time()
        draft_generation_time += draft_end - draft_start
        # tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")

        # print(f'output: {tokenizer.batch_decode(x)}')

        target_start = time.time()
        _ = target_model_cache.generate(x, 1)
        target_end = time.time()
        target_foward_time += target_end - target_start
        
        draft_forward_count += gamma
        
        n = prefix_len + gamma - 1
        
        # print(f"current N {n}")
        # return

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # print(f"shape of approx_model_cache {approx_model_cache._prob_history[:, prefix_len + i - 1, :].shape}")
                # collect_training_data(approx_model_cache._prob_history[:, prefix_len + i - 1, :],j,False,(approx_model_cache._prob_history[:, prefix_len + i - 1, j]))
                # reject
                n = prefix_len + i - 1
                break
            
            # if verbose:
            #     # print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            # collect_training_data(approx_model_cache._prob_history[:, prefix_len + i - 1, :],j,True,(approx_model_cache._prob_history[:, prefix_len + i - 1, j]))

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        # print(f"shape of target_model_cache._prob_history {target_model_cache._prob_history.shape}, approx_model_cache._prob_history {approx_model_cache._prob_history.shape}")
        if n < prefix_len + gamma - 1:
            
            target_dist = max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :])
            # target_dist = target_model_cache._prob_history[:, n, :] 
            # reject someone, sample from the pos n
            # t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            # if verbose:
            #     print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            target_dist = target_model_cache._prob_history[:, -1, :]
            # t = sample(target_model_cache._prob_history[:, -1, :])
            # if verbose:
            #     print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        #TODO: make the prefix multi-batch here. 
        # prefix = torch.cat((prefix, t), dim=1)

    
    #FIXME: the acceptance rate may be calculated wrong 
    # print(f"acceptance rate is {accepted_count/(prefix.shape[-1] - seq_len)} generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    stats = {'acceptance_rate':accepted_count/draft_forward_count,
             'target_forward_count':target_sample_count+resample_count,
             'draft_forward_count':draft_forward_count,
             'draft_generation_time':draft_generation_time,
             'target_foward_time':target_foward_time}
    # stats = None
    return prefix,stats

def beam_draft_generation(gamma, prefix, target_distribution,draft_model):
    # return a single prefix 
    # first repeat the prefix 
    # get the number of beam in return of determine_num_batch()
    
    
    top_k_prob, top_k_token = determine_num_batch(target_distribution,upper_bound=3)
    
    if draft_model._past_key_values != None:
        #FIXME: however, the repeat here, maybe expensive? 
        # print(f'Shape of draft_model._past_key_values is {draft_model._past_key_values.shape}')
        # draft_model._past_key_values = draft_model._past_key_values.repeat(top_k_token.shape[1], 1,1) 
        # 😢 give up the kv-cache, to check if acceptance rate is increase, fixed the kv-cache later
        draft_model._past_key_values = None
    replicated_input_ids = prefix.repeat(top_k_token.shape[1], 1)
    # print(f"shape of replicated_input_ids is {replicated_input_ids.shape}")
    d1,d2 = top_k_token.shape[0],top_k_token.shape[1]
    
    top_k_prob = top_k_prob.view((d2,d1))
    top_k_token = top_k_token.view((d2,d1))
    batched_prefix =torch.cat((replicated_input_ids,top_k_token),dim=-1)
    while True:
        if gamma > 0:
            gamma -=1
        else: 
            break 
        
        q = draft_model._forward_with_kvcache(batched_prefix, False)
        next_tok = sample(q)
    
        b,t_n = next_tok.shape[0],next_tok.shape[1]
        next_tok = next_tok.view(t_n,b)
        next_prob = q.gather(1,next_tok)
         
        batched_prefix = torch.cat((batched_prefix,next_tok),dim=-1)
        top_k_prob = torch.cat((top_k_prob,next_prob),dim=-1)
    # print(f"shape of grouped_prefix is {top_k_prob}")
    best_sequence_idx = beam_search_candidates(top_k_prob)
    # print(f"final sequence idx {best_sequence_idx}")
    picked_prefix = batched_prefix[best_sequence_idx,:]

    draft_model._prob_history = draft_model._prob_history[best_sequence_idx]
    picked_prefix = picked_prefix[None]
    # print(f'Shape of draft_model._past_key_values is {len(draft_model._past_key_values)}, shape is {draft_model._past_key_values[0][0].shape} len {len(draft_model._past_key_values[0])}, what is best_sequence_idx {best_sequence_idx} ')
    # draft_model._past_key_values = draft_model._past_key_values[:,:,best_sequence_idx]
    # print(f'Shape of draft_model._past_key_values after select index is {len(draft_model._past_key_values)}, shape is {draft_model._past_key_values[0]}')
    # draft_model._past_key_values = draft_model._past_key_values[None]
    draft_model._prob_history =  draft_model._prob_history[None]
    return picked_prefix

def beam_search_candidates(batch_prefixes):
    epsilon = 1e-9
    log_likelihood = torch.log(batch_prefixes + epsilon)
    
    summed_log_likelihood = torch.sum(log_likelihood, dim=-1)
    
    max_log_likelihood_index = torch.argmax(summed_log_likelihood)
    
    return max_log_likelihood_index.item()
def determine_num_batch(target_dist,upper_bound ):
    # need to dynamically deter mine the lower bound of k with eta sampling 
    top_k_probs, top_k_indices = torch.topk(torch.softmax(target_dist, dim=-1), upper_bound)

    # need be careful about the type of tokens and probs, and the device, and shape#FIXME:
    # not completely sure the top_k_indices is the token
    return top_k_probs,top_k_indices
