# Reference: 
import time 
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

from transformers import AutoTokenizer
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
# from inference.decision_models import  *

    
  
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

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

class TargetInitializedtMCSD:
    def __init__(
        self,
        draft_model,
        target_model,
        mcsd_config,
        eos_token_id,
        max_new_tokens: int = 200,
        temperature: float = 0, # target and draft using the same temperature
    ) -> None:
       
        self.mcsd_config= mcsd_config 
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device
        self.target_model_device = (
            target_model.model.get_input_embeddings().weight.device
        )
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_draft_len = len(mcsd_config)-1
        self.temperature = temperature
        self.target_past_key_values = None
        self.draft_past_key_values = None
    

        prod_size = torch.cumprod(torch.tensor(mcsd_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.divide_factor =[ prod_size[-1]//i for i in prod_size[1:]]
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(mcsd_config).to(
            device=self.draft_model_device
        )
        

        # stats collection
        self.stats = {'ground_acceptance_count': 0,
                       "draft_generation_time":0.0, 
                       "verification_time":0.0, 
                       "total_generation_round":0, 
                       "total_generated_draft_tokens":0}

    def topk_target_token(self,target_dist ):
        """
        use to sample multiple token from target distribution to perform target first mcsd
        """
        if self.temperature >0:
            cand_tokens = torch.multinomial(
                target_dist, self.mcsd_config[0], replacement=False
            ).view(1, -1)
        else:
            _, topk_index = target_dist.topk(
                        k=self.mcsd_config[0], dim=-1
                    )
            cand_tokens = topk_index.view(1, -1)
        return cand_tokens
    
    def generation_loop(self,
                        input_ids,):
        with torch.no_grad():
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
            batch_tokens = self.topk_target_token(torch.softmax(logits,dim=-1))
            multi_candid_token  = torch.cat((input_ids,batch_tokens), dim=1)
            self.target_past_key_values = list(target_output.past_key_values)  # target_output.past_key_values

            # using the draft model just to get the past_key_values
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
            self.stats['ground_acceptance_count'] = 0
            self.stats['total_generated_draft_tokens'] = 0


            while True:
                self.stats['total_generation_round'] += 1
                input_ids, cand_probs = self.generate_draft(multi_candid_token,init_input_len)
                # self.stats['total_generated_draft_tokens'] += input_ids.size(-1) - init_input_len
                self.stats['total_generated_draft_tokens'] += len(self.mcsd_config)-1

                verification_start = time.time()
                target_ground_prob, accepted_depth, input_ids =self.verification(input_ids,cand_probs)
                
                verification_end = time.time()
                self.stats['verification_time']+= verification_end - verification_start
                self.stats['ground_acceptance_count'] += accepted_depth.int()
                if (
                    self.eos_token_id in input_ids[0, -(accepted_depth+2) :]
                    or input_ids.size(-1) - non_change_input_len >= self.max_new_tokens
                ):
                    break
                init_input_len = input_ids.size(-1)
                batch_tokens = self.topk_target_token(target_ground_prob)
                multi_candid_token  = torch.cat((input_ids, batch_tokens), dim=1)
                
            return input_ids, self.stats
        
    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        init_input_length: int,
    ):
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        step_tree_attn_self_mask = None

        # assume the input ids in generate draft always have past key values
        pruned_input_ids = input_ids[:, self.draft_past_key_values[0][0].size(2) :]
        step_tree_attn_mask = None
        position_ids = None
        # -----------------------------------------------------------------------
        ## I think I need to be careful here because the kvcache here is considering th
        # draft model, not the target model 
        for s in range(self.max_draft_len):
            
            step = s +1
            step_k = self.mcsd_config[step]
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
                #⭐ when the last generation is full accepted, and draft kv cache is not enough
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
                # need to skip prefix and target sampled token 
                hidden_states = hidden_states[0,-self.mcsd_config[0]:]
            else:
                hidden_states = hidden_states[0]


            logits = self.draft_model.lm_head(hidden_states)  # seq_len x hidden_dim
            past_key_values = list(outputs.past_key_values)
            self.draft_past_key_values = past_key_values

            if self.temperature == 0:
                # greedy
                topk_logit, topk_index = logits.topk(
                    k=step_k, dim=-1
                )  # seq_len x k
                topk_probs = torch.softmax(topk_logit, dim=-1)
                step_cand_probs = torch.zeros_like(logits)
                step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                cand_tokens = topk_index.view(1, -1)
            else:
                step_cand_probs = torch.softmax(logits / self.temperature, dim=-1)
                cand_tokens = torch.multinomial(
                    step_cand_probs, step_k, replacement=False
                ).view(1, -1)
            cand_probs.append(step_cand_probs)

            draft_generation_end = time.time()
            generate_draft_time = draft_generation_end- draft_generation_start
            self.stats['draft_generation_time'] += generate_draft_time
            pruned_input_ids = cand_tokens
            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)
        
        return input_ids,cand_probs

    def _forward_target_model(
        self,
        input_ids: torch.LongTensor,
    ):
        input_ids = input_ids.to(self.target_model_device)
        # the tree attn len here need to be change to shape of maximum accepted attn shape + 1
        tree_attn_len = self.tree_attn_self_mask.size(0)
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
            tree_attn_mask[-tree_attn_len:, -tree_attn_len:] = self.tree_attn_self_mask
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
            tree_attn_mask[:, init_input_length:] = self.tree_attn_self_mask
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
    
    ## old
    def match_and_stack_tensors(self,tensor_list):
        # Get the shape of the last tensor in the list as the target shape
        target_shape = tensor_list[-1].shape[0]
        
        # Process each tensor to match the target shape using repeat_interleave
        expanded_tensors = []
        for tensor in tensor_list:
            # Calculate the repeat factor
            repeat_factor = target_shape // tensor.shape[0]
            
            # Use repeat_interleave to expand the tensor
            expanded_tensor = torch.repeat_interleave(tensor, repeats=repeat_factor, dim=0)
            
            # Add the expanded tensor to the list
            expanded_tensors.append(expanded_tensor)
        
        # Stack all tensors along the first dimension
        result = torch.cat(expanded_tensors, dim=0)
        return result
    


    def repeat_tokens(self, token_tensor,is_token = True):
        # old with repeat
        # Calculate the number of tokens at each layer

        # Calculate the total number of tokens
        # total_tokens = sum(num_tokens)
        # assert total_tokens == token_tensor.size(0), "Token tensor size does not match total tokens required."

        # Calculate the repeats for each layer
        if is_token:
            # calculate for token to prepare for square tensor
            num_tokens = self.prod_size[2:]
            last_layer_tokens = num_tokens[-1]
            repeats_for_layer = [last_layer_tokens // n for n in num_tokens]
        else:
            # calculate for ground probability 
            num_tokens = self.prod_size[1:]
            repeats_for_layer = self.mcsd_config[1:]

        # Generate the repeats array for the entire tensor
        repeats = []
        for count, repeat in zip(num_tokens, repeats_for_layer):
            repeats.extend([repeat] * count)

        repeats = torch.tensor(repeats, dtype=torch.long).to(self.target_model_device)

        # Use torch.repeat_interleave to modify the tensor
        modified_tensor = torch.repeat_interleave(token_tensor, repeats, dim=0)
        return modified_tensor
    # def repeat_tokens(self, token_tensor):
    #     # new with indexing
    #     num_tokens = self.prod_size[1:]
    #     total_tokens = sum(num_tokens)
    #     assert total_tokens == token_tensor.size(0), "Token tensor size does not match total tokens required."
        
    #     # Compute the repeat factors without actually repeating the tensor
    #     last_layer_tokens = num_tokens[-1]
    #     repeats_for_layer = [last_layer_tokens // n for n in num_tokens]
        
    #     # Compute indices to simulate the repeated tensor
    #     indices = []
    #     start = 0
    #     for count, repeat in zip(num_tokens, repeats_for_layer):
    #         end = start + count
    #         indices.extend([i for i in range(start, end) for _ in range(repeat)])
    #         start = end
        
    #     indices = torch.tensor(indices, dtype=torch.long, device=token_tensor.device)
        
    #     # Use indexing instead of repeat_interleave
    #     modified_tensor = token_tensor[indices]
    #     return modified_tensor
    
    def verification(
            self,
            input_ids,
            candidate_probs
    ):
        # new verification
        dim = len(candidate_probs)
        input_ids = input_ids.to(self.target_model_device)
        logits = self._forward_target_model(input_ids)
        logits = logits[0]
        init_input_length = input_ids.size(1) -self.tree_attn_self_mask.size(0)
        keep_indices = list(range(init_input_length))
        draft_keep_indices = keep_indices
        if self.temperature > 0:
            ground_probs = torch.softmax(logits / self.temperature,dim=-1)
        else:
            _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
            ground_probs = torch.zeros_like(logits)
            ground_probs.scatter_(dim=1, index=topk_index, value=1)
        token_to_verify = input_ids[:,init_input_length+ self.mcsd_config[0]:]

        if self.temperature >0:
            modified_ground_probs = self.repeat_tokens(ground_probs[:-self.prod_size[-1],:],is_token=False)
            candidate_probs = torch.cat(candidate_probs,dim=0)
            modified_candidate_probs = self.repeat_tokens(candidate_probs,is_token=False)
            draft_probs_mask= torch.gather(modified_candidate_probs,dim=-1,index = token_to_verify).squeeze(-1)
            ground_probs_mask = torch.gather(modified_ground_probs,dim=-1,index = token_to_verify).squeeze(-1)
            ground_over_cand = ground_probs_mask/draft_probs_mask

            # 😅 need a new way to repeat the probability? 
            accept_prob = torch.rand(1,ground_over_cand.size(-1),device=self.target_model_device)
            posterior_mask = (ground_over_cand>=accept_prob).int()
        else:
            # this should be greedy 
            ground_argmax = torch.argmax(ground_probs[:-self.prod_size[-1]],dim=-1)
            modified_ground_probs = self.repeat_tokens(ground_argmax,is_token=False) 
            posterior_mask = (token_to_verify==modified_ground_probs).int()
        posterior_mask = self.repeat_tokens(posterior_mask.squeeze(0)).view(dim,self.prod_size[-1])
        candidate_accept_length = (torch.cumprod(posterior_mask,dim=0)).sum(dim=0)
        longest_length = candidate_accept_length.max()
        # next_multi_tokens_logits = ground_probs[0]
        if longest_length == 0:
            # all rejected 
            next_multi_tokens_logits = ground_probs[0]

            # this logits should be the last multi-token's target logits 
            keep_indices.append(init_input_length)# the logits immediately after input, 
        else: 
            
            longest_candidate_index = torch.argmax(candidate_accept_length).item()
            # if longest_candidate_index>9:
            #     print(f"check the longest_candidate_index {longest_candidate_index} and the mask is {posterior_mask}")
            # self.max_token_path[longest_candidate_index].append(longest_length)
            # print(f"check longest_candidate_index {longest_candidate_index} ")

            # +1 due to cps start with 0, + 1 due to depth is for draft -1 due to index start with 0
            ground_index = self.cumulative_prod_size[longest_length] + longest_candidate_index//self.divide_factor[longest_length]
            
            #TODO: find new index
            if longest_length == self.max_draft_len:
                next_multi_tokens_logits = ground_probs[ground_index]
            else:
                if self.temperature > 0:
                    # normalized the distribution to restore the target distribution 
                    # the candidate prob need -1 because the depth of candidate prob is always one less than logits, due to target first 
                    diff = ground_probs[ground_index] - candidate_probs[ground_index] 
                    diff = torch.nn.functional.relu(diff,inplace=True)
                    diff /=diff.sum()
                    next_multi_tokens_logits = diff
                else:
                    # greedy
                    next_multi_tokens_logits = ground_probs[ground_index]

            
            # plus one here because need consider the target made tokens 
            for depth in range(longest_length+1):
                # using cumulative prod size and divide factor to map the square index to tree expansion index
                keep_indices.append(self.cumulative_prod_size[depth] + longest_candidate_index//self.divide_factor[depth]+init_input_length) 
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )
        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        if longest_length == self.max_draft_len:
            draft_keep_indices = keep_indices[: len(keep_indices)-1]
        else:
            draft_keep_indices = keep_indices
        self.update_kv_cache(target_kv=keep_indices,draft_kv=draft_keep_indices)
        
        return next_multi_tokens_logits, longest_length, input_ids
    
    # def verification(
    #         self,
    #         input_ids,
    #         candidate_probs
    # ):
    #     dim = len(candidate_probs)
    #     input_ids = input_ids.to(self.target_model_device)
    #     logits = self._forward_target_model(input_ids)
    #     logits = logits[0]
    #     init_input_length = input_ids.size(1) -self.tree_attn_self_mask.size(0)
    #     keep_indices = list(range(init_input_length))
    #     draft_keep_indices = keep_indices
    #     if self.temperature > 0:
    #         ground_probs = torch.softmax(logits / self.temperature,dim=-1)
    #     else:
    #         _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
    #         ground_probs = torch.zeros_like(logits)
    #         ground_probs.scatter_(dim=1, index=topk_index, value=1)

    #     repeat_ground_probs = self.repeat_tokens(ground_probs)
    #     repeat_input_ids = self.repeat_tokens(input_ids[:,init_input_length:].squeeze(0)).unsqueeze(0)
    #     reshaped_ground_probs = repeat_ground_probs.view(-1,self.prod_size[-1],logits.size(-1))
    #     ground_probs_to_verify = repeat_ground_probs.view(-1,self.prod_size[-1],logits.size(-1))
    #     token_to_verify =repeat_input_ids[:,self.prod_size[-1]:].view(-1,self.prod_size[-1]).unsqueeze(-1)

    #     if self.temperature >0:
    #         candidate_probs = self.match_and_stack_tensors(candidate_probs).view(dim,self.prod_size[-1],-1)
    #         draft_probs_mask= torch.gather(candidate_probs,dim=-1,index = token_to_verify).squeeze(-1)
    #         ground_probs_mask = torch.gather(ground_probs_to_verify,dim=-1,index = token_to_verify).squeeze(-1)
    #         ground_over_cand = ground_probs_mask/draft_probs_mask

    #         # 😅 need a new way to repeat the probability? 
    #         accept_prob = torch.rand(dim,self.prod_size[-1],device=self.target_model_device)
    #         posterior_mask = (ground_over_cand>=accept_prob).int()
    #     else:
    #         # this should be greedy 
            
    #         posterior_mask = (token_to_verify.squeeze(-1)==torch.argmax(ground_probs_to_verify[:-1,:,:],dim=-1)).int()

    #     candidate_accept_length = (torch.cumprod(posterior_mask,dim=0)).sum(dim=0)
    #     longest_length = candidate_accept_length.max()
    #     if longest_length == 0:
    #         # all rejected 
    #         next_multi_tokens_logits = ground_probs[0]

    #         # this logits should be the last multi-token's target logits 
    #         keep_indices.append(init_input_length)# the logits immediately after input, 
    #     else: 
            
    #         longest_candidate_index = torch.argmax(candidate_accept_length).item()
    #         # self.max_token_path[longest_candidate_index].append(longest_length)

    #         # print(f"check longest_candidate_index {longest_candidate_index} ")
    #         if longest_length == self.max_draft_len:
    #             next_multi_tokens_logits = reshaped_ground_probs[longest_length,longest_candidate_index]
    #         else:
    #             if self.temperature > 0:
    #                 # normalized the distribution to restore the target distribution 
    #                 # the candidate prob need -1 because the depth of candidate prob is always one less than logits, due to target first 
    #                 diff = reshaped_ground_probs[longest_length,longest_candidate_index] - candidate_probs[longest_length-1,longest_candidate_index] 
    #                 diff = torch.nn.functional.relu(diff,inplace=True)
    #                 diff /=diff.sum()
    #                 next_multi_tokens_logits = diff
    #             else:
    #                 # greedy
    #                 next_multi_tokens_logits = reshaped_ground_probs[longest_length,longest_candidate_index]

            
    #         # plus one here because need consider the target made tokens 
    #         for depth in range(longest_length+1):
    #             # using cumulative prod size and divide factor to map the square index to tree expansion index
    #             keep_indices.append(self.cumulative_prod_size[depth] + longest_candidate_index//self.divide_factor[depth]+init_input_length) 
    #     keep_indices = torch.tensor(
    #         keep_indices, dtype=torch.long, device=self.target_model_device
    #     )
    #     input_ids = input_ids.index_select(dim=1, index=keep_indices)
    #     # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    #     if longest_length == self.max_draft_len:
    #         draft_keep_indices = keep_indices[: len(keep_indices)-1]
    #     else:
    #         draft_keep_indices = keep_indices
    #     self.update_kv_cache(target_kv=keep_indices,draft_kv=draft_keep_indices)
        
    #     return next_multi_tokens_logits, longest_length, input_ids

    