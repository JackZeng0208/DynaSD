import time 
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import pandas as pd
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

target_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(target_model_name)

"""
this function should return the maximum dynamic mask 
for example like 10 width 10 depth which is 100 tokens 
and the following program will trim the mask with indices 
"""
def get_tree_attn_self_mask(max_config: Tuple[int]):
    max_config = torch.tensor(max_config, dtype=torch.int)
    prod_size = torch.cumprod(max_config, dim=0)
    mask_size = prod_size.sum().item()
    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    attn_mask = attn_mask.diagonal_scatter(torch.ones(mask_size))
    # run BFS
    idx_queue = [
        (0, None, idx) for idx in list(range(max_config[0]))
    ]  # each node: (depth, parent, idx)
    while len(idx_queue) != 0:
        depth, parent, idx = idx_queue.pop(0)
        if parent is not None:
            attn_mask[idx, : parent + 1] = attn_mask[parent, : parent + 1]

        if depth != len(max_config) - 1:
            idx_base = prod_size[:depth].sum().item()
            child_idx_base = prod_size[: depth + 1].sum().item()
            for child_idx_bias in range(max_config[depth + 1]):
                real_child_idx = (
                    (idx - idx_base) * max_config[depth + 1]
                    + child_idx_base
                    + child_idx_bias
                )
                idx_queue.append((depth + 1, idx, real_child_idx))
    return attn_mask

class decision_model(nn.Module):
    def __init__(self):
        super(decision_model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class NewTreeStrategy:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        config_width = 6,
        config_dpeth = 10,
        using_decision_model = True,
        max_new_tokens: int = 200,
        decision_threshold = 0.5,
        greedy = False,
        
        draft_model_temp: float = 0,
        target_model_temp: float = 0,
        decision_model_path: str = "/home/iasl-transformers/UCI-IASL-Transformer/decision_model/decision_model.pth",
    ) -> None:
        """
        so the max_config here could be 5 in width and 10 in depth 
        """
        self.greedy = greedy
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.max_config = self.generate_fork_config(width=config_width,depth=config_dpeth)
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device
        self.target_model_device = (
            target_model.model.get_input_embeddings().weight.device
        )
        self.max_draft_len = len(self.max_config)-1
        self.draft_model_temp = draft_model_temp
        self.target_model_temp = target_model_temp
        self.target_pastjkey_values = None
        self.draft_past_key_values = None

        prod_size = torch.cumprod(torch.tensor(self.max_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(self.max_config).to(
            device=self.draft_model_device
        )

        ## for decision model 
        self.using_decision_model = using_decision_model
        self.decision_model = decision_model()
        self.decision_model.load_state_dict(torch.load(decision_model_path))
        self.decision_model.cuda()
        self.decision_model.eval()
        self.stop_generation = False
        self.continue_depth = 0 
        self.decision_threshold = decision_threshold
        self.dynamic_target_mask =None

        # stats collection
        
        self.stats = {'token/second': 0.0,'ground_acceptance_count': 0,
                       "draft_generation_time":0.0, "verification_time":0.0, 
                       "total_generation_round":0, "decision_model_time":0.0, 
                       "decision_acceptance_count":0}
    
    def generate_fork_config(self,width,depth):
        config = []
        config.append(width)
        for _ in range(depth):
            config.append(1)
        return config
    
    
    def determine_num_batch(self,target_dist ):
        # need to dynamically deter mine the lower bound of k with eta sampling 
        # top_k_probs, top_k_indices = torch.topk(torch.softmax(target_dist, dim=-1), upper_bound)
        topk_logit, topk_index = target_dist.topk(
                    k=self.max_config[0], dim=-1
                )
        cand_tokens = topk_index.view(1, -1)
        # not completely sure the top_k_indices is the token
        return torch.softmax(topk_logit, dim=-1),cand_tokens
    

    def generation_loop(self,
                        input_ids,):
        input_ids = input_ids.to(self.draft_model_device)
        non_change_input_len = input_ids.size(-1)
        init_input_len = input_ids.size(-1)
        target_output = self.target_model.model(
            input_ids = input_ids,
            use_cache=True,
            past_key_values=None,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            position_ids=None,
        )
        hidden_states = target_output.last_hidden_state
        hidden_states= hidden_states[0,-1:]
        logits = self.target_model.lm_head(hidden_states)
        # topk_logit, topk_index = logits.topk(
        #             k=self.max_config[0], dim=-1
        #         )
        # cand_tokens = topk_index.view(1, -1)
        target_probs, batch_tokens = self.determine_num_batch(logits)
        multi_candid_token  = torch.cat((input_ids,batch_tokens), dim=1)


        self.target_past_key_values = list(target_output.past_key_values)  # target_output.past_key_values


        # using the draft model just to get the past_key_values
        # TODO: however there is better way of modifying the attention mask , refer the 
        # init step of target forward
        draft_output = self.draft_model.model(
            input_ids = input_ids,
            use_cache=True,
            past_key_values=None,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            position_ids=None,
        )
        self.draft_past_key_values = list(draft_output.past_key_values)
        # pruned_input_ids = multi_candid_token[:, self.draft_past_key_values[0][0].size(2) :]
        while True:
            self.stats['total_generation_round'] += 1

            draft_model_start = time.time()
            input_ids, cand_probs = self.generate_draft(multi_candid_token,init_input_len)
            draft_model_end = time.time()
            self.stats['draft_generation_time'] += draft_model_end - draft_model_start
            # print(f"input ids shape is {input_ids.shape}, self.continue depth is {self.continue_depth}")
            # return
            verification_start = time.time()
            if not self.greedy:
                target_ground_prob, accepted_depth, input_ids =self.speculative_sampling(input_ids,cand_probs)
            else:
                target_ground_prob, accepted_depth, input_ids = self.greedy_verify(input_ids)
            verification_end = time.time()
            self.stats['verification_time']+= verification_end - verification_start
            self.stats['decision_acceptance_count'] += self.continue_depth
            self.stats['ground_acceptance_count'] += accepted_depth
            if (
                self.eos_token_id in input_ids[0, -(self.continue_depth+1) :]
                or input_ids.size(-1) - non_change_input_len >= self.max_new_tokens
            ):
                break
            # 🤔 change latter, hot fix for current cuda error 
            # if (input_ids.size(-1) - non_change_input_len >= self.max_new_tokens):
            #     break
            init_input_len = input_ids.size(-1)
            target_probs, batch_tokens = self.determine_num_batch(target_ground_prob)
            multi_candid_token  = torch.cat((input_ids, batch_tokens), dim=1)
            
        end = time.time()
        # stats['token/second'] = (input_ids.size(-1) - non_change_input_len)/(end - start)
        return input_ids, self.stats


    def continue_decision_check(self,entropy,prob):
        """
        1. this will return a bool on whether need further generation 
        the model taking probs and entropy, so this function 
        dont calculate the entropy or copy the logit to save time
        input(entropy, prob)
        """
        cat_input = torch.cat((entropy.view(-1,1),prob.view(-1,1)),dim = -1) # 🤔 the dimension matters here 
        with torch.no_grad():
            output = self.decision_model(cat_input)
        result = torch.any(output > self.decision_threshold).item()
        if result:
            return True
        return False

        
    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        init_input_length: int,
    ):
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        # pruned_input_ids = None
        """
        right here I need to call the target model to generate dist for first 
        multicandidate generation 
        """
        step_tree_attn_self_mask = None
        

        # the input ids in generate draft always have past key values
        pruned_input_ids = input_ids[:, self.draft_past_key_values[0][0].size(2) :]
        ## 🤔may be i don't need to worry about the keycache for target here? 
        ## or I should because ignoring the kv cache here will potentially slow down
        ## the generation 
        # TODO: the k here should be dynamically picked 
        step_tree_attn_mask = None
        position_ids = None
        self.continue_depth = 0
        # -----------------------------------------------------------------------
        ## I think I need to be careful here because the kvcache here is considering th
        # draft model, not the target model 
        #TODO: the for loop should replace by the decision model 
        for s in range(self.max_draft_len):
            
            step = s +1
            step_k = 1 # since this is fork like then only 1 is needed 
            if self.stop_generation: 
                # at least one draft generation 
                self.stop_generation = False
                self.dynamic_target_mask = self.tree_attn_self_mask[:self.cumulative_prod_size[step],:self.cumulative_prod_size[step]].clone()
                return input_ids,tuple(cand_probs)
                

            self.continue_depth +=1
            step_tree_attn_self_mask = self.tree_attn_self_mask[
                self.cumulative_prod_size[step - 1] : self.cumulative_prod_size[
                    step],: self.cumulative_prod_size[step],] 
            position_ids = torch.full(
                    (1, self.prod_size[step]),
                    init_input_length + step - 1,
                    dtype=torch.long,
                    device=self.draft_model_device,)
            context_attn_mask = torch.ones(
                    (self.prod_size[step], init_input_length), dtype=torch.bool
                ).to(self.draft_model_device)
            step_tree_attn_mask = torch.cat(
                    (context_attn_mask, step_tree_attn_self_mask), dim=1
                )
            if pruned_input_ids.size(-1) > self.prod_size[step]:
                # when the last generation is full accepted, and draft kv cache is not enough
                last_non_kv_cached_id = torch.tensor([init_input_length-1], dtype=torch.long, device=self.draft_model_device)[None]
                position_ids = torch.cat((last_non_kv_cached_id, position_ids), dim=1)
                new_row = torch.ones(1, step_tree_attn_mask.size(1), dtype=torch.bool).to(self.draft_model_device)
                num_ones = (step_tree_attn_mask[0] == 1).sum() - 1
                new_row[0, num_ones:] = 0
                step_tree_attn_mask = torch.cat((new_row, step_tree_attn_mask), dim=0)
                

            draft_generation_start = time.time()
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=self.draft_past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                tree_attn_mask=step_tree_attn_mask,
                position_ids=position_ids,
            )

            hidden_states = outputs.last_hidden_state
            # originally the step is 0, but in my case the step start from 1
            if step == 1:
                # 🤔 may need to check here later TODO:
                hidden_states = hidden_states[0,-self.max_config[0]:]
            else:
                hidden_states = hidden_states[0]
            logits = self.draft_model.lm_head(hidden_states)  # seq_len x hidden_dim

            past_key_values = list(outputs.past_key_values)
            self.draft_past_key_values = past_key_values

            if self.draft_model_temp == 0:
                topk_logit, topk_index = logits.topk(
                    k=step_k, dim=-1
                )  # seq_len x k
                topk_probs = torch.softmax(topk_logit, dim=-1)
                step_cand_probs = torch.zeros_like(logits)
                step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                cand_tokens = topk_index.view(1, -1)
            else:
                step_cand_probs = torch.softmax(logits / self.draft_model_temp, dim=-1)
                cand_tokens = torch.multinomial(
                    step_cand_probs, step_k, replacement=True
                ).view(1, -1)
            cand_probs.append(step_cand_probs)
            
            entropy = torch.distributions.Categorical(probs=step_cand_probs).entropy()
            max_probs = torch.max(step_cand_probs,dim=-1).values
            draft_generation_end = time.time()
            generate_draft_time = draft_generation_end- draft_generation_start
            
            decision_model_start = time.time()
            if self.using_decision_model and self.continue_decision_check(entropy=entropy,prob=max_probs) == False:
                self.stop_generation = True
            decision_model_end = time.time()
            generate_decision_model_time = decision_model_end - decision_model_start

            self.stats['draft_generation_time'] += generate_draft_time
            self.stats['decision_model_time'] += generate_decision_model_time

            pruned_input_ids = cand_tokens

            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)
        self.dynamic_target_mask = self.tree_attn_self_mask.clone()

        return input_ids,cand_probs

    def _forward_target_model(
        self,
        input_ids: torch.LongTensor,
    ):
        input_ids = input_ids.to(self.target_model_device)
        # the tree attn len here need to be change to shape of maximum accepted attn shape + 1
        tree_attn_len = self.dynamic_target_mask.size(0)
        init_input_length = input_ids.size(1) - tree_attn_len
        init_forward = False

        if self.target_past_key_values is not None:
            pruned_input_ids = input_ids[:, self.target_past_key_values[0][0].size(2):]
        else:
            pruned_input_ids = input_ids
            init_forward = True

        if init_forward:
            tree_attn_mask = torch.zeros(
                (input_ids.size(1), input_ids.size(1)),
                dtype=torch.bool,
                device=self.target_model_device,
            )
            mask_cond = torch.arange(
                tree_attn_mask.size(-1), device=self.target_model_device
            )
            tree_attn_mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(tree_attn_mask.size(-1), 1), 1
            )
            tree_attn_mask[-tree_attn_len:, -tree_attn_len:] = self.dynamic_target_mask
            position_ids = tree_attn_mask.sum(dim=1) - 1

        else:
            tree_attn_mask = torch.ones(
                (
                    tree_attn_len, # I don't need plus 1 here
                    input_ids.size(1),
                ), 
                dtype=torch.bool,
                device=self.target_model_device,
            )
            # print(f"shape of tree attn mask is {tree_attn_mask.int()}")
            #⭐ I need change the self.tree attn self mask here to the new mask 
            tree_attn_mask[:, init_input_length:] = self.dynamic_target_mask
            # print(f"shape of tree attn mask after dynamid target mask  {tree_attn_mask.int()}")
            # tree_attn_mask[0, init_input_length:] = 0 # 🤔 this row may yielding incorrect position ids
            position_ids = tree_attn_mask.sum(dim=1) - 1
        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=self.target_past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        self.target_past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(
            hidden_states[:, -tree_attn_len - 1 :]
        )  # 1 x seq_len x hidden_din
        return logits
    
    def update_kv_cache(self,target_kv,draft_kv):
        
        for i in range(len(self.target_past_key_values)):
            self.target_past_key_values[i] = (
                self.target_past_key_values[i][0].index_select(
                    dim=2, index=target_kv
                ),
                self.target_past_key_values[i][1].index_select(
                    dim=2, index=target_kv
                ),
            )
        for i in range(len(self.draft_past_key_values)):
            self.draft_past_key_values[i] = (
                self.draft_past_key_values[i][0].index_select(
                    dim=2, index=draft_kv
                ),
                self.draft_past_key_values[i][1].index_select(
                    dim=2, index=draft_kv
                ),
            )


    def greedy_verify(
        self,
        input_ids: torch.LongTensor,
    ):
        
        input_ids = input_ids.to(self.target_model_device)
        logits = self._forward_target_model(input_ids)
        logits = logits[0]
        # think about how to dynamic decide the logits needed 
        # first reshape the input ids into 2d matrix 
        # logtis need first 20, while token need last 20 
        init_input_length = input_ids.size(1) -self.dynamic_target_mask.size(0)
        keep_indices = list(range(init_input_length))
        draft_keep_indices = keep_indices
        # dim0 is depth, dim1 is number of candidate
        reshaped_logits = logits.view(-1,self.max_config[0],logits.size(-1))

        # no need for the last 5
        logits_to_verify = logits[:-self.max_config[0],:].view(-1,self.max_config[0],logits.size(-1))
        token_to_verify = input_ids[:,init_input_length+self.max_config[0]:].view(-1,self.max_config[0])
        # logits_to_verify = logits_to_verify.transpose()
        
        # print(f"verify input ids {input_ids[:,init_input_length+self.max_config[0]:]}")
        # print(f"token to verify to check the dim of reshape {token_to_verify}")
        posterior_mask = (token_to_verify == torch.argmax(logits_to_verify,dim=-1)).int()
        # if self.max_config[0] != 1:
            # single candidate don't need transpose for horizontal verification 
        posterior_mask = torch.transpose(posterior_mask,0,1)
        # print(f"check posterior_mask {posterior_mask}")
        candidate_accept_length = (torch.cumprod(posterior_mask,dim=-1)).sum(dim=-1)
        longest_length  = candidate_accept_length.max()
        # print(f"check longest_length {longest_length} ")
        # print(f'what is wrong with longest_length {longest_length}, and self.max_config {self.max_config} and self.max_draf_length {self.max_draft_len}')
        if longest_length == 0:
            # all rejected 
            next_multi_tokens_logits = logits[0]
            # this logits should be the last multi-token's target logits 
            keep_indices.append(init_input_length)# the logits immediately after input, 
        else: 
            
            longest_candidate_index = torch.argmax(candidate_accept_length)
            # print(f"check longest_candidate_index {longest_candidate_index} ")
            next_multi_tokens_logits = reshaped_logits[longest_length,longest_candidate_index]
            
            # plus one here because need consider the target made tokens 
            for depth in range(longest_length+1):
                keep_indices.append(depth * (self.max_config[0]) + longest_candidate_index+init_input_length) # minus one here because the index start from 0
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )
        # print(f"check if keep_indices have redundant {keep_indices}")
        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        if longest_length == self.continue_depth:
            draft_keep_indices = keep_indices[: len(keep_indices)-1]
        else:
            draft_keep_indices = keep_indices
        self.update_kv_cache(target_kv=keep_indices,draft_kv=draft_keep_indices)
        
        return next_multi_tokens_logits, longest_length, input_ids
    

    def speculative_sampling(self,
                             input_ids,
                             candidate_probs):
        """
        1. first prepare the candiate probs as a mask 
        2. create a same shape mask from target model's logits 
        then do the something in greedy verify 
        """
        # print(f"shape of candidate_probs {len(candidate_probs)}")
        dim = len(candidate_probs)
        candidate_probs = torch.stack(candidate_probs,dim=0)
        #device=self.target_model_device)
        # print(f"what is the shape of candidate probs {candidate_probs.shape}, and device {candidate_probs.device} ")
        input_ids = input_ids.to(self.target_model_device)
        logits = self._forward_target_model(input_ids)
        logits = logits[0]
        ground_probs = torch.softmax(logits,dim=-1)
        # print(f"shape of logits is {logits.shape}")
        # think about how to dynamic decide the logits needed 
        # first reshape the input ids into 2d matrix 
        # logtis need first 20, while token need last 20 
        init_input_length = input_ids.size(1) -self.dynamic_target_mask.size(0)
        keep_indices = list(range(init_input_length))
        draft_keep_indices = keep_indices
        # dim0 is depth, dim1 is number of candidate
        reshaped_logits = ground_probs.view(-1,self.max_config[0],logits.size(-1))
        # no need the last layer logits for verification 
        ground_probs_to_verify = ground_probs[:-self.max_config[0],:].view(-1,self.max_config[0],logits.size(-1))
        token_to_verify = input_ids[:,init_input_length+self.max_config[0]:].view(-1,self.max_config[0])
    
        draft_probs_mask = candidate_probs[torch.arange(dim)[:,None,None],
                                           torch.arange(self.max_config[0])[None,:,None],
                                           token_to_verify[:,:,None]].squeeze(-1)
        ground_probs_mask = ground_probs_to_verify[torch.arange(dim)[:,None,None],
                                           torch.arange(self.max_config[0])[None,:,None],
                                           token_to_verify[:,:,None]].squeeze(-1)
    
        ground_over_cand = ground_probs_mask/draft_probs_mask
        accept_prob = torch.rand(dim,self.max_config[0],device=self.target_model_device)

        # print(f"what is the shape of accpet_prob {accept_prob}")
        # print(f"what is ground_over_cand {ground_over_cand}")
        posterior_mask = (ground_over_cand>=accept_prob).int().transpose(0,1)
        candidate_accept_length = (torch.cumprod(posterior_mask,dim=-1)).sum(dim=-1)
        longest_length  = candidate_accept_length.max()
        # print(f"check longest_length {longest_length} ")
        # print(f'what is wrong with longest_length {longest_length}, and self.max_config {self.max_config} and self.max_draf_length {self.max_draft_len}')
        if longest_length == 0:
            # all rejected 
            next_multi_tokens_logits = logits[0]
            # this logits should be the last multi-token's target logits 
            keep_indices.append(init_input_length)# the logits immediately after input, 
        else: 
            
            longest_candidate_index = torch.argmax(candidate_accept_length)
            # print(f"check longest_candidate_index {longest_candidate_index} ")
            if longest_length == self.continue_depth:
                next_multi_tokens_logits = reshaped_logits[longest_length,longest_candidate_index]
            else:
                # normalized the distribution to restore the target distribution 
                # ⭐ the candidate prob need -1 because the depth of candidate prob is always one less than logits, due to target first 
                diff = reshaped_logits[longest_length,longest_candidate_index] - candidate_probs[longest_length-1,longest_candidate_index] 
                diff_max = torch.where(diff>0,diff,torch.zeros_like(diff))
                diff_max_sum = torch.sum(diff_max,dim=-1,keepdim=True)
                next_multi_tokens_logits = diff_max/diff_max_sum
            
            # plus one here because need consider the target made tokens 
            for depth in range(longest_length+1):
                keep_indices.append(depth * (self.max_config[0]) + longest_candidate_index+init_input_length) # minus one here because the index start from 0
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )
        # print(f"check if keep_indices have redundant {keep_indices}")
        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        if longest_length == self.continue_depth:
            draft_keep_indices = keep_indices[: len(keep_indices)-1]
        else:
            draft_keep_indices = keep_indices
        self.update_kv_cache(target_kv=keep_indices,draft_kv=draft_keep_indices)
        
        return next_multi_tokens_logits, longest_length, input_ids
