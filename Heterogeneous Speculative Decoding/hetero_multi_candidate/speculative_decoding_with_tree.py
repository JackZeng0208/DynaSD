import torch
import time
import zmq
from utils import sample, norm_logits, max_fn, KVCacheModel

from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    """
    Base class for draft outputs of decoder-only generation models using speculative decoding.
    add tree_config since server side do not know tree configuration
    """

    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None
    tree_config: Optional[Tuple] = None

@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    """
    Base class for verification outputs of decoder-only generation models using speculative decoding.
    not using kv-cache for server side
    """
    sequences: torch.LongTensor = None
    draft_model_accept_indices: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None


class hetero_speculative_decoding:
    def __init__(self, stats: bool = False):
        """
        Args:
            stats (bool, optional): whether to print stats. Defaults to True.
        """
        self.time_spend_sending_message = 0
        self.time_spend_on_draft_model_generation = 0
        self.time_spend_on_target_model_forward = 0
        self.stats = stats

    def get_time_spend_on_draft_model_generation(self):

        return self.time_spend_on_draft_model_generation

    def get_time_spend_on_target_model_forward(self):
        return self.time_spend_on_target_model_forward

    def get_time_spend_sending_message(self):
        return self.time_spend_sending_message

    

    # function contain the communitcation between server and edge
    def edge_tree_attn_speculative_decoding(
        self,
        input_ids,
        draft_model,
        server_ip,
        max_len,
        tree_config = (2,2,1),
        temperature = 1,
        client_id = ""):
        """
        1. need tree_truncation_function in future
        2. 
        """
        edge_draft_generator = Edge_side_tree_strategy_generation(
            draft_model= draft_model,
            use_tree_trunction=False,
            tree_config=tree_config,
        )

        seq_len = input_ids.shape[1]
        T = seq_len + max_len

        resample_count = 0
        target_sample_count = 0
        accepted_count = 0
        total_draft_generate_count = 0
        input_ids = input_ids.to(edge_draft_generator.draft_model_device)
        start_time = time.time()
        draft_tokens = None

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server_ip}:1919")

        while input_ids.shape[1] < T:
            prefix_len = input_ids.shape[1]

            draft_generate_start_time = time.time()
            """
            sequences=input_ids,
                past_key_values=past_key_values,
                cand_probs=tuple(cand_probs),
                tree_config = self.tree_config)
            """
            output = edge_draft_generator.generate_draft_naive_tree_attn(
                input_ids=input_ids,
                past_key_values= edge_draft_generator.draft_model_past_key_values,
            )
            draft_generate_end_time = time.time()

            draft_tokens, _ , cand_probs, draf_tree_config = output ## naive tree, dynamic tree also need tree_config
            total_draft_generate_count += len(edge_draft_generator.tree_config)
            self.time_spend_on_draft_model_generation += draft_generate_end_time - \
                draft_generate_start_time

            # Send draft tokens to server
            send_tensor_start_time = time.time()
            socket.send_pyobj(
                {'draft_tokens': draft_tokens, 
                'client_id': client_id, 
                'tree_config':draf_tree_config,
                'cand_probs': cand_probs}) ## cand probs needed for naive tree_attn
            target_model_mesg_dict = socket.recv_pyobj()
            send_tensor_end_time = time.time()

            """
            1. TODO: get the updated input_ids from server side
            2. update the draft model kv-cache
            3. 
            """
            new_tokens = target_model_mesg_dict['new_tokens'] ## verified token + new token sampled from target
            accepted_indices = target_model_mesg_dict['accepted_indices'] ## to update draft model kv cache
            accepted_count += target_model_mesg_dict['accept_count'] ## update stats
            edge_draft_generator.update_kv_cache(accepted_indices)
            # target_model_history = target_model_mesg_dict['target_prob_hist']
            target_model_generation_time = target_model_mesg_dict['target_model_generation_time']
            total_time_in_server = target_model_generation_time
            self.time_spend_sending_message += send_tensor_end_time - \
                send_tensor_start_time - total_time_in_server
            self.time_spend_on_target_model_forward += target_model_generation_time

            """
            TODO:
            1. first naively change input_ids to new tokens,
            2. later need consider a efficient way, 
            ex. store prefix tokens in candidate, and cat the new accept tokens. 
            """
            input_ids = new_tokens.to(edge_draft_generator.draft_model_device)
            print(f"shape of input_ids.shape[1]: {input_ids.shape[1]}")
        if self.stats:
            print(f"generated tokens numbers {input_ids.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        end_time = time.time()
        print(f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
        print(f"Token Generation Speed (with speculative decoding): {max_len / (end_time - start_time)} tokens/s")
        print(f"Acceptance Rate: {accepted_count / total_draft_generate_count}")
        return input_ids

    def server_tree_attn_speculative_decoding(
                                    self,
                                    socket,
                                    target_model: torch.nn.Module,
                                    temperature: float = 1,):
        """
        1. TODO: optimization, not creating server class every time call this function? 
        will that save some space or time ?
        """
        server_verifier = Server_side_verification(
            target_model = target_model)

        draft_tokens_dict = {}
        draft_tokens = None
        while True:
            message = socket.recv_pyobj()
            client_id = message['client_id']
            received_draft_tokens = message['draft_tokens']
            cand_probs = message['cand_probs']
            draft_tree_config = message['tree_config']
            draft_tokens_dict[client_id] = received_draft_tokens
            draft_tokens = draft_tokens_dict[client_id]
            draft_tokens = draft_tokens.to(server_verifier.target_model_device)
            target_forward_time = time.time()
            target_logits = server_verifier.target_forward(draft_tokens)
            finish_target_forward_time = time.time()

            verification_time = time.time()
            output = server_verifier.verify_longest_candidate_hetero(
                input_ids=draft_tokens,
                cand_probs=cand_probs,
                tree_config= draft_tree_config)
            
            new_tokens = output.sequences
            accepted_indices = output.draft_model_accept_indices
            accept_count = output.acceptance_count
            draft_tokens_dict[client_id] = None
            response = {
                'new_tokens': new_tokens,
                'accepted_indices': accepted_indices,
                'accept_count': accept_count,
                'target_model_generation_time': finish_target_forward_time - target_forward_time,
                'client_id': client_id
            }
            socket.send_pyobj(response)


    def edge_speculative_decoding(self,
                                  input_ids: torch.Tensor,
                                  draft_model: torch.nn.Module,
                                  server_ip: str,
                                  max_len: int,
                                  gamma: int = 4,
                                  temperature: float = 1,
                                  top_k: int = 0,
                                  top_p: float = 0,
                                  client_id: str = "") -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): input tensor
            draft_model (torch.nn.Module): draft model for speculative decoding
            server_ip (str): server IP
            max_len (int): maximum length of token generation 
            gamma (int, optional): gamma. Defaults to 4.
            temperature (float, optional): temperature. Defaults to 1.
            top_k (int, optional): top k. Defaults to 0.
            top_p (float, optional): top p. Defaults to 0.
            random_seed (int, optional): random seed. Defaults to 1234.
            client_id (str, optional): client ID. Defaults to None.
        """
        draft_model.to('cuda:0')
        seq_len = input_ids.shape[1]
        T = seq_len + max_len
        approx_model_cache = KVCacheModel(
            draft_model, temperature, top_k, top_p).to('cuda:0')

        resample_count = 0
        target_sample_count = 0
        accepted_count = 0
        input_ids = input_ids.to('cuda:0')
        start_time = time.time()
        draft_tokens = None

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server_ip}:1919")

        while input_ids.shape[1] < T:
            prefix_len = input_ids.shape[1]

            draft_generate_start_time = time.time()
            draft_tokens = approx_model_cache.generate(
                input_ids, gamma)
            draft_generate_end_time = time.time()
            self.time_spend_on_draft_model_generation += draft_generate_end_time - \
                draft_generate_start_time

            # Send draft tokens to server
            send_tensor_start_time = time.time()
            socket.send_pyobj(
                {'draft_tokens': draft_tokens, 'client_id': client_id})
            target_model_mesg_dict = socket.recv_pyobj()
            send_tensor_end_time = time.time()

            target_model_history = target_model_mesg_dict['target_prob_hist']
            target_model_generation_time = target_model_mesg_dict['target_model_generation_time']
            total_time_in_server = target_model_generation_time
            self.time_spend_sending_message += send_tensor_end_time - \
                send_tensor_start_time - total_time_in_server
            self.time_spend_on_target_model_forward += target_model_generation_time

            target_model_history = target_model_history.to('cuda:0')

            n = prefix_len + gamma - 1
            for i in range(gamma):
                r = torch.rand(1, device='cuda:0')
                j = draft_tokens[:, prefix_len + i]
                if r > (target_model_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break
                accepted_count += 1

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            input_ids = draft_tokens[:, :n + 1]
            approx_model_cache.rollback(n + 1)
            assert approx_model_cache._prob_history.shape[-2] <= n + \
                1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"

            if n < prefix_len + gamma - 1:
                t = sample(max_fn(
                    target_model_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
                resample_count += 1
            else:
                assert n == target_model_history.shape[1] - 1
                t = sample(target_model_history[:, -1, :])
                target_sample_count += 1

            input_ids = input_ids.to("cuda:0")
            input_ids = torch.cat((input_ids, t), dim=1)

        if self.stats:
            print(f"generated tokens numbers {input_ids.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        end_time = time.time()
        print(f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
        print(f"Token Generation Speed (with speculative decoding): {max_len / (end_time - start_time)} tokens/s")
        print(f"Acceptance Rate: {accepted_count / max_len}")
        approx_model_cache.clear_cache()
        return input_ids

    @torch.no_grad()
    def sampling_without_kvcache(self,
                                 draft_tokens: torch.Tensor,
                                 target_model: torch.nn.Module,
                                 temperature: float = 1,
                                 top_k: int = 0,
                                 top_p: float = 0) -> list:
        """
        Args:
            draft_tokens (torch.Tensor): tokens generated by draft model
            target_model (torch.nn.Module): target model for speculative decoding
            temperature (float, optional): Defaults to 1.
            top_k (int, optional): Defaults to 0.
            top_p (float, optional): Defaults to 0.
            random_seed (int, optional): Defaults to 1234.
        """
        target_model_history = target_model(draft_tokens).logits
        for i in range(target_model_history.shape[-2]):
            target_model_history[:, i, :] = norm_logits(
                target_model_history[:, i, :], temperature, top_k, top_p)
        return target_model_history

    def server_speculative_decoding(self,
                                    socket,
                                    target_model: torch.nn.Module,
                                    temperature: float = 1,
                                    top_k: int = 0,
                                    top_p: float = 0,
                                    random_seed: int = 1234
                                    ):
        """
        Args:
            socket (zmq.Socket): zmq socket object used for communication
            target_model (torch.nn.Module): target model for speculative decoding
        """
        draft_tokens_dict = {}
        draft_tokens = None
        target_model.to("cuda:0")
        while True:
            message = socket.recv_pyobj()
            client_id = message['client_id']
            received_draft_tokens = message['draft_tokens']
            draft_tokens_dict[client_id] = received_draft_tokens
            draft_tokens = draft_tokens_dict[client_id]
            draft_tokens = draft_tokens.to("cuda:0")
            target_forward_time = time.time()
            target_model_history_tensor = self.sampling_without_kvcache(
                draft_tokens=draft_tokens,
                target_model=target_model,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            finish_target_forward_time = time.time()
            draft_tokens_dict[client_id] = None
            response = {
                'target_prob_hist': target_model_history_tensor,
                'target_model_generation_time': finish_target_forward_time - target_forward_time,
                'client_id': client_id
            }
            socket.send_pyobj(response)

class Server_side_verification:
    def __init__(
        self,
        target_model,
        target_model_temp = 1,):
        """
        1. have no information about tree configuration 
        """
        self.target_model = target_model
        self.target_model_temp = target_model_temp
        self.target_model_device = target_model.model.get_input_embeddings().weight.device
    
    def target_forward(self,input_ids):
        input_ids = input_ids.to(self.target_model_device)
        tree_attn_len = self.tree_attn_self_mask.size(0)
        init_input_length = input_ids.size(1) - tree_attn_len
        pruned_input_ids = input_ids

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


        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        

        logits = self.target_model.lm_head(
            hidden_states[:, -tree_attn_len - 1 :]
        )  # 1 x seq_len x hidden_dim
        return logits

    def verify_longest_candidate_hetero(
        self,
        input_ids: torch.LongTensor,        
        cand_probs: Optional[Tuple[torch.FloatTensor]],
        tree_config: Tuple = (2,2,1) 
    ) -> DecoderOnlyVerificationOutput:

        ## prepare for heterogeneous 
        self.cand_probs = cand_probs
        self.tree_config = tree_config
        self.max_draft_len = len(self.tree_config)
        self.total_num_path = int(torch.prod(torch.tensor(self.tree_config)).item())
        self.total_path = [[] for _ in range(self.total_num_path)] # for picking the longest path
        prod_size = torch.cumprod(torch.tensor(self.tree_config, dtype=torch.int,device=self.target_model_device), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        input_ids = input_ids.to(self.target_model_device)
        logits, target_model_past_key_values = self._forward_target_model(
            input_ids, target_model_past_key_values
        )
        logits = logits[0]  # seq_len x hidden_dim
        tree_attn_len = self.tree_attn_self_mask.size(0)
        self.unverified_tokens = input_ids[0, -tree_attn_len:]
        self.init_input_length = input_ids.size(1) - tree_attn_len

        # use sampling no greedy. 
        ground_probs = torch.softmax(logits / self.target_model_temp, dim=-1)
        # print(f"shape of ground_probs {ground_probs.shape}")
        keep_indices = list(range(self.init_input_length))
        to_drop_len = 0

        current_ground_prob = [ground_probs[0]]
        init_ground_prob = ground_probs[0] # prepare for no candidate accepted
        ground_probs = ground_probs[1:]
        idx_group_bias = [0]
        cand_probs_idx = [0]
        for depth in range(self.max_draft_len):
            current_ground_prob, idx_group_bias,cand_probs_idx  = self.verify_single_layer(
                depth = depth,
                ground_probs=ground_probs,
                idx_group_biases= idx_group_bias,
                verification_probs_list=current_ground_prob,
                current_layer_cand_prob_idx = cand_probs_idx
            )
            if len(current_ground_prob)  == 0 :

                break 
        # may want to consider a tie breaker for keep_indices 
        if len(current_ground_prob) == 0:
            tail_ground_prob = init_ground_prob
            # means all candidate rejected
        else: 
            
            longest_list_index = find_longest_list_index(self.total_path)
            if len(self.total_path[longest_list_index]) == self.max_draft_len:
                to_drop_len= 1
                depth = self.max_draft_len
            keep_indices.extend(self.total_path[longest_list_index])
            tail_ground_prob = ground_probs[self.total_path[longest_list_index][-1] - self.init_input_length]
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )

        # the to_drop_len is necessary here
        if to_drop_len != 0:
            draft_keep_indices = keep_indices[: len(keep_indices) - to_drop_len]
        else:
            draft_keep_indices = keep_indices
        
        
        tail_ground_token = torch.multinomial(tail_ground_prob, num_samples=1).to(
            device=input_ids.device
        )
        print(f"debug 329: what is tail_ground_token[None] {tail_ground_token[None]}, and tail_ground_token is {tail_ground_token}")
        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        input_ids = torch.cat((input_ids, tail_ground_token[None]), dim=1)

        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            draft_model_accept_indices = draft_keep_indices,
            acceptance_count=depth,
        )

    def verify_single_layer(self,
        depth:int,
        idx_group_biases:List, 
        verification_probs_list:List,
        current_layer_cand_prob_idx,
        ground_probs):
        """
        idx_group_bias and ground_probs should have the same size - []
        """
        next_layer_ground_probs = []
        next_layer_idx_group_biases = []
        next_layer_cand_idx = []
        for i in range(len(verification_probs_list)):
            current_ground_prob = verification_probs_list[i]
            idx_group_bias = idx_group_biases[i]
            cand_probs_idx  = current_layer_cand_prob_idx[i]

            idx_base = self.cumulative_prod_size[depth] + idx_group_bias
            accept_idx_biases = self.new_verification(
                current_ground_prob,
                self.cand_probs[depth][cand_probs_idx],
                self.unverified_tokens[idx_base : idx_base + self.tree_config[depth]],
            )
            if len(accept_idx_biases) != 0:
                for accept_idx_bias in accept_idx_biases:
                    global_idx = idx_base + accept_idx_bias
                    next_layer_ground_probs.append(ground_probs[global_idx])
                    # update self.total_path

                    self.update_total_path(idx=self.init_input_length + global_idx,
                    idx_in_heap= global_idx,
                    depth=depth)

                    # handle keep_indices after get the full list
                
                    if depth < self.max_draft_len - 1:
                        cand_probs_idx = idx_group_bias + accept_idx_bias
                        next_layer_cand_idx.append(cand_probs_idx)
                        next_layer_idx_group_biases.append(cand_probs_idx * self.tree_config[depth + 1])
        return  next_layer_ground_probs,  next_layer_idx_group_biases, next_layer_cand_idx
    
    def update_total_path(self,
        idx, 
        idx_in_heap ,
        depth, ):

        repeat = 1
        if depth < self.max_draft_len -1:
            repeat = self.tree_config[depth +1]
        # print(f"what is self_prod size {self.prod_size}")
        k = self.prod_size[depth+1]
        p = self.cumulative_prod_size[depth +1]
        # print(f'idx is {idx} and idx_in_heap is {idx_in_heap}, k is {k}, p is {p}, depth is {depth}, repeat is {repeat}')
        first_idx = k - (p-idx_in_heap)
        offset = first_idx*repeat
        for _ in range(repeat):
            self.total_path[offset].append(idx)
            offset+=1


class Edge_side_tree_strategy_generation:
    def __init__(
        self,
        draft_model,
        use_tree_trunction = False, # not implemented yet
        tree_config = (2,2,1),
        draft_model_temp = 1,
        
                ):
        # assume the model is on device
        self.use_tree_trunction = use_tree_trunction
        self.draft_model = draft_model,
        self.draft_model_temp = draft_model_temp
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device

        # if passed in tree_config, considered this as traditional tree-attention with fixed configuration
        self.tree_config = tree_config
        prod_size = torch.cumprod(torch.tensor(self.tree_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(self.tree_config).to(
            device=self.draft_model_device
        )
        self.draft_model_past_key_values = None
    
    def update_kv_cache(self,indices):
        if self.draft_model_past_key_values != None:
            for i in range(len(self.draft_model_past_key_values)):
                draft_model_past_key_values[i] = (
                    draft_model_past_key_values[i][0].index_select(
                        dim=2, index=indices
                    ),
                    draft_model_past_key_values[i][1].index_select(
                        dim=2, index=indices
                    ),
                )
        
        
    def generate_draft_naive_tree_attn(
        self,
        input_ids, # already in device from its caller
        past_key_values, # for kv cache 
    ):
        pass
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        init_input_length = input_ids.size(1)
        max_draft_len = len(tree_config)

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = input_ids
        for step in range(max_draft_len):
            if step != 0:
                step_tree_attn_self_mask = self.tree_attn_self_mask[
                    self.cumulative_prod_size[step - 1] : self.cumulative_prod_size[
                        step
                    ],
                    : self.cumulative_prod_size[step],
                ]
                position_ids = torch.full(
                    (1, self.prod_size[step]),
                    init_input_length + step - 1,
                    dtype=torch.long,
                    device=self.draft_model_device,
                )
                context_attn_mask = torch.ones(
                    (self.prod_size[step], init_input_length), dtype=torch.bool
                ).to(self.tree_attn_self_mask)
                step_tree_attn_mask = torch.cat(
                    (context_attn_mask, step_tree_attn_self_mask), dim=1
                )
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                tree_attn_mask=step_tree_attn_mask,
                position_ids=position_ids,
            )
            hidden_states = outputs.last_hidden_state

            if step == 0:
                hidden_states = hidden_states[0, -1:]
            else:
                hidden_states = hidden_states[0]
            logits = self.draft_model.lm_head(hidden_states)  # seq_len x hidden_dim

            past_key_values = list(outputs.past_key_values)
            ## TODO: not sure if I can update kv like this 
            self.draft_model_past_key_values = past_key_values

            step_cand_probs = torch.softmax(logits / self.draft_model_temp, dim=-1)
            cand_tokens = torch.multinomial(
                step_cand_probs, step_k, replacement=False
            ).view(1, -1)
            cand_probs.append(step_cand_probs)
            pruned_input_ids = cand_tokens
            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)
            
            return DecoderOnlyDraftOutput(
                sequences=input_ids,
                past_key_values=past_key_values,
                cand_probs=tuple(cand_probs),
                tree_config = self.tree_config)

    def generate_draft_dynamic_tree_attn(
        self
    ):
        """
        1. generate k during draft_generation 
        2. return tokens and dynamic tree configuration 
        """
        pass 