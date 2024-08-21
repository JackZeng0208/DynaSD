import torch
from tqdm import tqdm
import torch

from .kvcache_model import KVCacheModel
from .utils import norm_logits, sample, max_fn

@torch.no_grad()
def greedy_speculative_sampling_for_training(
                        prefix : torch.Tensor,
                        approx_model : torch.nn.Module,
                        target_model : torch.nn.Module, 
                        max_len : int, 
                        gamma : int = 1,
                        temperature : float = 1, 
                        top_k : int = 0,
                        top_p : float = 0,
                        verbose : bool = False,
                        random_seed : int = None) -> torch.Tensor:
    
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert approx_model.device == target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    total_generation_count = 0
    train_data = []
    labels = []
    
    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        total_generation_count += gamma

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            
            j = x[:, prefix_len + i]
            target_token = torch.argmax(target_model_cache._prob_history[:, prefix_len + i - 1, :],dim=-1)
            draft_top_prob = approx_model_cache._prob_history[:prefix_len+1-1,:].topk(k=10,dim=-1)
            entropy = torch.distributions.Categorical(probs=approx_model_cache._prob_history[:prefix_len+1-1,:]).entropy()
            train_data.append(torch.cat(entropy,draft_top_prob),dim=0)
            

            if target_token != j:
                # reject
                n = prefix_len+i-1
                labels.append(0)
                break
            else:
                labels.append(1)
            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            # t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            t = sample(target_model_cache._prob_history[:, n, :])
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix, train_data, labels
   

@torch.no_grad()
def greedy_speculative_sampling(
                        prefix : torch.Tensor,
                        approx_model : torch.nn.Module,
                        target_model : torch.nn.Module, 
                        max_len : int, 
                        gamma : int = 4,
                        temperature : float = 1, 
                        top_k : int = 0,
                        top_p : float = 0,
                        verbose : bool = False,
                        random_seed : int = None) -> torch.Tensor:
    
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert approx_model.device == target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    total_generation_count = 0
    
    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        total_generation_count += gamma

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            
            j = x[:, prefix_len + i]
            target_token = torch.argmax(target_model_cache._prob_history[:, prefix_len + i - 1, :],dim=-1)
            if target_token != j:
                # reject
                n = prefix_len+i-1
                break
            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            # t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            t = sample(target_model_cache._prob_history[:, n, :])
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix, accepted_count / total_generation_count



@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
   
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
    total_generation_count = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        total_generation_count += gamma

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix, accepted_count / total_generation_count