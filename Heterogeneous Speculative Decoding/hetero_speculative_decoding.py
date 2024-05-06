import torch
import time
import zmq
from utils import sample, norm_logits, max_fn, KVCacheModel
from typing import Callable, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

import pynvml
import threading
import csv
pynvml.nvmlInit()


def get_tree_attn_self_mask(k_config: Tuple[int]):
    """
    1. need to generate tree attention mask for naive tree attention
    2. not accept k_config for dynamic tree
    """
    k_config = torch.tensor(k_config, dtype=torch.int)
    prod_size = torch.cumprod(k_config, dim=0)
    mask_size = prod_size.sum().item()
    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    attn_mask = attn_mask.diagonal_scatter(torch.ones(mask_size))
    # run BFS
    idx_queue = [
        (0, None, idx) for idx in list(range(k_config[0]))
    ]  # each node: (depth, parent, idx)
    while len(idx_queue) != 0:
        depth, parent, idx = idx_queue.pop(0)
        if parent is not None:
            attn_mask[idx, : parent + 1] = attn_mask[parent, : parent + 1]

        if depth != len(k_config) - 1:
            idx_base = prod_size[:depth].sum().item()
            child_idx_base = prod_size[: depth + 1].sum().item()
            for child_idx_bias in range(k_config[depth + 1]):
                real_child_idx = (
                    (idx - idx_base) * k_config[depth + 1]
                    + child_idx_base
                    + child_idx_bias
                )
                idx_queue.append((depth + 1, idx, real_child_idx))
    return attn_mask


def find_longest_list_index(nested_list):
    """ 
    1. use to find which candidate sequence is longest 
    TODO: may need to use torch alternative for optimization
    """
    longest_length = 0
    longest_index = None

    for index, lst in enumerate(nested_list):
        if len(lst) > longest_length:
            longest_length = len(lst)
            longest_index = index

    return longest_index


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


class HeteroSpeculativeDecoding:
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
            edge_socket,
            max_len,
            tree_config=(3, 2, 1),
            temperature=1,
            client_id=""):
        """
        1. need tree_truncation_function in future
        2. currently using predefined k_config
        3. process: 
            1. generate token 
            2. send token and candidate probs to server for verify and update
            3. update the input_ids and kv cache from server's message
        """
        edge_draft_generator = EdgeSideTreeStrategyGeneration(
            draft_model=draft_model,
            use_tree_trunction=False,
            tree_config=tree_config,
        )

        seq_len = input_ids.shape[1]
        T = seq_len + max_len

        resample_count = 0
        target_sample_count = 0
        accepted_count = 0
        total_draft_generate_count = 0
        verification_time = 0
        input_ids = input_ids.to(edge_draft_generator.draft_model_device)
        start_time = time.time()
        draft_tokens = None
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
                past_key_values=edge_draft_generator.draft_model_past_key_values,
                draft_model=draft_model,
            )
            draft_generate_end_time = time.time()

            draft_tokens = output.sequences
            cand_probs = output.cand_probs
            # print(f"cand_probs in edge {cand_probs}")
            # naive tree, dynamic tree also need tree_config
            draf_tree_config = output.tree_config
            total_draft_generate_count += len(edge_draft_generator.tree_config)
            self.time_spend_on_draft_model_generation += draft_generate_end_time - \
                draft_generate_start_time

            # Send draft tokens to server
            # print(f"line 155: the type of draft_token before sending should be tensor {draft_tokens}")
            send_tensor_start_time = time.time()
            edge_socket.send_pyobj(
                {'draft_tokens': draft_tokens,
                'client_id': client_id,
                'tree_config': draf_tree_config,
                'cand_probs': cand_probs})  # cand probs needed for naive tree_attn
            target_model_mesg_dict = edge_socket.recv_pyobj()
            send_tensor_end_time = time.time()

            """
            1. TODO: get the updated input_ids from server side
            2. update the draft model kv-cache
            3. 
            """
            new_tokens = target_model_mesg_dict['new_tokens']  # verified token + new token sampled from target
            # to update draft model kv cache
            accepted_indices = target_model_mesg_dict['accepted_indices']
            # update stats
            accepted_count += target_model_mesg_dict['accept_count']
            verification_time += target_model_mesg_dict['verification_time']
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

        if self.stats:
            print(f"generated tokens numbers {input_ids.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        end_time = time.time()

        print(
            f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
        print(
            f"Token Generation Speed (with speculative decoding): {max_len / (end_time - start_time)} tokens/s")
        print(
            f"Acceptance Rate: {accepted_count / total_draft_generate_count}")
        print(f"Total verification time is {verification_time}")
        torch.cuda.empty_cache()
        return input_ids, accepted_count / total_draft_generate_count

    def server_tree_attn_speculative_decoding(
            self,
            server_socket,
            target_model: torch.nn.Module,
            temperature: float = 1,):
        """
        1. TODO: optimization, not creating server class every time call this function? 
        will that save some space or time ?
        
        process:
            1. get token and cand_probs from edge
            2. forward get ground true probs for verify
            3. use reject sample (need cand_probs and ground truth) to verify
            4. return the new tokens and accepted index back to edge device, and some stats variable
        """
        server_verifier = ServerSideVerification(
            target_model=target_model)

        draft_tokens_dict = {}
        draft_tokens = None
        device = torch.device("cuda:0")
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
        with open(f"gpu_utilization_MCSD.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            while True:
                gpu_utilization = []
                def capture_gpu_utilization(stop_event):
                    # Adjust the sample interval as needed (in seconds) -> 1ms
                    sample_interval = 0.01
                    while not stop_event.is_set():
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        gpu_utilization.append(utilization)
                        time.sleep(sample_interval)
                # Start capturing GPU utilization in a separate thread
                stop_event = threading.Event()
                gpu_thread = threading.Thread(target=capture_gpu_utilization, args=(stop_event,))
                gpu_thread.start()

                message = server_socket.recv_pyobj()
                client_id = message['client_id']
                received_draft_tokens = message['draft_tokens']
                # print(f"line 215: the type of draft_token just received from edge should be tensor {draft_tokens}")

                cand_probs = message['cand_probs']
                draft_tree_config = message['tree_config']
                draft_tokens_dict[client_id] = received_draft_tokens
                draft_tokens = draft_tokens_dict[client_id]
                # print(f"line 221: the type of draft_token after dict should be tensor {draft_tokens}")

                draft_tokens = draft_tokens.to(server_verifier.target_model_device)
                target_forward_time = time.time()
                target_logits = server_verifier.target_forward(
                    draft_tokens, tree_config=draft_tree_config)
                finish_target_forward_time = time.time()

                verification_time = time.time()
                output = server_verifier.verify_longest_candidate_hetero(
                    input_ids=draft_tokens,
                    cand_probs=cand_probs,
                    logits=target_logits,)
                end_verification_time = time.time()

                new_tokens = output.sequences
                accepted_indices = output.draft_model_accept_indices
                accept_count = output.acceptance_count
                draft_tokens_dict[client_id] = None
                response = {
                    'new_tokens': new_tokens,
                    'accepted_indices': accepted_indices,
                    'accept_count': accept_count,
                    'target_model_generation_time': finish_target_forward_time - target_forward_time,
                    'client_id': client_id,
                    'verification_time': end_verification_time - verification_time
                }
                server_socket.send_pyobj(response)

                # Stop capturing GPU utilization
                stop_event.set()
                gpu_thread.join()
                writer.writerow([gpu_utilization])

    def edge_speculative_decoding(self,
                                  input_ids: torch.Tensor,
                                  draft_model: torch.nn.Module,
                                  edge_socket: zmq.Socket,
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
        total_draft_generate_count = 0

        while input_ids.shape[1] < T:
            prefix_len = input_ids.shape[1]

            draft_generate_start_time = time.time()
            draft_tokens = approx_model_cache.generate(
                input_ids, gamma)
            total_draft_generate_count += gamma
            draft_generate_end_time = time.time()
            self.time_spend_on_draft_model_generation += draft_generate_end_time - \
                draft_generate_start_time

            # Send draft tokens, prob_history, prefix_len, and gamma to server
            send_tensor_start_time = time.time()
            edge_socket.send_pyobj(
                {'draft_tokens': draft_tokens, 'prob_history': approx_model_cache._prob_history, 'prefix_len': prefix_len, 'gamma': gamma, 'client_id': client_id})
            target_model_mesg_dict = edge_socket.recv_pyobj()
            send_tensor_end_time = time.time()

            input_ids = target_model_mesg_dict['input_ids']
            accepted_count += target_model_mesg_dict['accepted_count']
            resample_count += target_model_mesg_dict['resample_count']
            target_sample_count += target_model_mesg_dict['target_sample_count']
            total_time_in_server = target_model_mesg_dict['total_time_in_server']
            rollback_num = target_model_mesg_dict['rollback']
            
            self.time_spend_sending_message += send_tensor_end_time - \
                send_tensor_start_time - total_time_in_server
            self.time_spend_on_target_model_forward += total_time_in_server

            approx_model_cache.rollback(rollback_num)

        if self.stats:
            print(
                f"generated tokens numbers {input_ids.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        end_time = time.time()
        print(
            f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
        print(
            f"Token Generation Speed (with speculative decoding): {max_len / (end_time - start_time)} tokens/s")
        print(f"Acceptance Rate: {accepted_count / total_draft_generate_count}")
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
                                server_socket,
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
        received_prob_history = {}
        draft_tokens = None
        target_model.to("cuda:0")
        
        device = torch.device("cuda:0")
        handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
        with open(f"gpu_utilization_vanilla_sd.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            while True:
                gpu_utilization = []
                def capture_gpu_utilization(stop_event):
                    # Adjust the sample interval as needed (in seconds) -> 1ms
                    sample_interval = 0.1
                    while not stop_event.is_set():
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        gpu_utilization.append(utilization)
                        time.sleep(sample_interval)
                # Start capturing GPU utilization in a separate thread
                stop_event = threading.Event()
                gpu_thread = threading.Thread(target=capture_gpu_utilization, args=(stop_event,))
                gpu_thread.start()

                message = server_socket.recv_pyobj()
                client_id = message['client_id']
                received_draft_tokens = message['draft_tokens']
                received_prob_history = message['prob_history']
                prefix_len = message['prefix_len']
                gamma = message['gamma']
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

                # Probability comparison (verification) on the server side
                accepted_count = 0
                resample_count = 0
                target_sample_count = 0
                n = prefix_len + gamma - 1
                for i in range(gamma):
                    r = torch.rand(1, device='cuda:0')
                    j = draft_tokens[:, prefix_len + i]
                    if r > (target_model_history_tensor[:, prefix_len + i - 1, j]) / (received_prob_history[:, prefix_len + i - 1, j]):
                        n = prefix_len + i - 1
                        break
                    accepted_count += 1

                input_ids = draft_tokens[:, :n + 1]

                if n < prefix_len + gamma - 1:
                    t = sample(max_fn(
                        target_model_history_tensor[:, n, :] - received_prob_history[:, n, :]))
                    resample_count += 1
                else:
                    assert n == target_model_history_tensor.shape[1] - 1
                    t = sample(target_model_history_tensor[:, -1, :])
                    target_sample_count += 1

                input_ids = torch.cat((input_ids, t), dim=1)
                draft_tokens_dict[client_id] = None

                response = {
                    'input_ids': input_ids,
                    'accepted_count': accepted_count,
                    'resample_count': resample_count,
                    'target_sample_count': target_sample_count,
                    'total_time_in_server': finish_target_forward_time - target_forward_time,
                    'client_id': client_id,
                    'rollback': n+1
                }
                server_socket.send_pyobj(response)

                # Stop capturing GPU utilization
                stop_event.set()
                gpu_thread.join()
                writer.writerow([gpu_utilization])

class ServerSideVerification:
    def __init__(
            self,
            target_model,
            target_model_temp=1,):
        """
        1. have no information about tree configuration 
        """
        self.target_model = target_model
        self.target_model_temp = target_model_temp
        self.target_model_device = target_model.model.get_input_embeddings().weight.device

    def new_verification(self,
                         ground_probs: torch.FloatTensor,
                         cand_probs: Tuple[torch.FloatTensor],
                         cand_tokens: torch.LongTensor,
                         ) -> Optional[int]:
        """
        previous verification may not return the longest sequence
        1. this version will return the longest sequence, but this is not typical acceptance 
        """
        accepted_indices = []
        cand_probs = cand_probs.to(ground_probs.device)
        for check_idx, cand_token in enumerate(cand_tokens):
            accept_threshold = ground_probs[cand_token] / \
                cand_probs[cand_token]
            if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
                accepted_indices.append(check_idx)
            else:
                # FIXME: why does this do?
                ground_probs -= cand_probs
                ground_probs = torch.nn.functional.relu(ground_probs, inplace=True)
                ground_probs /= ground_probs.sum()
        return accepted_indices

    def target_forward(self, input_ids,
                       tree_config):
        """ 
        1. generate the tree_attn mask and forward to get ground truth logit
        2. TODO: not considering target model kv cache right may want to have it in the future
        3. this can be used with dynamic tree 
        """
        self.tree_config = tree_config
        self.tree_attn_self_mask = get_tree_attn_self_mask(self.tree_config).to(
            device=self.target_model_device)
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
        tree_attn_mask[-tree_attn_len:, -
                       tree_attn_len:] = self.tree_attn_self_mask
        position_ids = tree_attn_mask.sum(dim=1) - 1

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.target_model.lm_head(
            hidden_states[:, -tree_attn_len - 1:]
        )  # 1 x seq_len x hidden_dim
        return logits

    def verify_longest_candidate_hetero(
        self,
        input_ids: torch.LongTensor,
        cand_probs: Optional[Tuple[torch.FloatTensor]],
        logits,
    ) -> DecoderOnlyVerificationOutput:
        """
        1. assume target forward is always called before verification 
        2. therefore the tree_config is initialize in forward
        """
        # prepare for heterogeneous
        self.cand_probs = cand_probs
        # print(f"line 499 what is the tree_config {self.tree_config}")
        self.max_draft_len = len(self.tree_config)
        self.total_num_path = int(torch.prod(
            torch.tensor(self.tree_config)).item())
        # for picking the longest path
        self.total_path = [[] for _ in range(self.total_num_path)]
        prod_size = torch.cumprod(torch.tensor(
            self.tree_config, dtype=torch.int, device=self.target_model_device), dim=0)
        prod_size = torch.cat(
            (torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        input_ids = input_ids.to(self.target_model_device)
        # logits, target_model_past_key_values = self._forward_target_model(
        #     input_ids, target_model_past_key_values
        # )
        logits = logits[0]  # seq_len x hidden_dim
        tree_attn_len = self.tree_attn_self_mask.size(0)
        self.unverified_tokens = input_ids[0, -tree_attn_len:]
        self.init_input_length = input_ids.size(1) - tree_attn_len

        # print('-'*50)
        # print(f"line 518: check for nan distribution, logit is {logits}\nshape of logits {logits.shape}\n")
        # print(f"line 519: check for invalid value in dist, max {torch.max(logits,-1)},\nmin {torch.min(logits,-1)}\n")
        # use sampling no greedy.
        # TODO: what if there is nan in the ground_prob, how will it affect the verification?
        # FIXME: when and why does ground truth probability has nan?
        ground_probs = F.softmax(logits / (self.target_model_temp), dim=-1)
        # print(f"shape of ground_probs {ground_probs.shape}")
        keep_indices = list(range(self.init_input_length))
        to_drop_len = 0

        current_ground_prob = [ground_probs[0]]
        init_ground_prob = ground_probs[0]  # prepare for no candidate accepted
        tail_ground_index = 0  # prepare for softmax produce nan case, use greedy argmax instead
        ground_probs = ground_probs[1:]
        idx_group_bias = [0]
        cand_probs_idx = [0]
        
        """ 
        1. currently is using reject sampling to verify
        2. can add conditional statement here to switch to typical acceptance, to get accepted token's index
        """
        for depth in range(self.max_draft_len):
            current_ground_prob, idx_group_bias, cand_probs_idx = self.verify_single_layer(
                depth=depth,
                ground_probs=ground_probs,
                idx_group_biases=idx_group_bias,
                verification_probs_list=current_ground_prob,
                current_layer_cand_prob_idx=cand_probs_idx
            )
            if len(current_ground_prob) == 0:

                break
        # print(f"line 540: total path is {self.total_path}")
        # may want to consider a tie breaker for keep_indices
        if len(current_ground_prob) == 0:

            tail_ground_prob = init_ground_prob
            # print(f"***all rejected use initial probs {tail_ground_prob}\n")
            # means all candidate rejected
        else:

            longest_list_index = find_longest_list_index(self.total_path)
            if len(self.total_path[longest_list_index]) == self.max_draft_len:
                to_drop_len = 1
                depth = self.max_draft_len
            keep_indices.extend(self.total_path[longest_list_index])
            tail_ground_index = self.total_path[longest_list_index][-1] - \
                self.init_input_length
            tail_ground_prob = ground_probs[tail_ground_index]
            # print(f"**accepted_some use  probs {tail_ground_prob}\n")
        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )

        # the to_drop_len is necessary here
        if to_drop_len != 0:
            draft_keep_indices = keep_indices[: len(
                keep_indices) - to_drop_len]
        else:
            draft_keep_indices = keep_indices

        # tail_ground_prob = torch.softmax(tail_ground_prob,dim=-1)
        """ 
        TODO: temporary solution for nan, may need better way to not greedy sample when tail probs has nan 
        """
        if (torch.isnan(tail_ground_prob).any()):
            tail_ground_token = torch.argmax(logits[tail_ground_index]).to(
                device=input_ids.device)
            tail_ground_token = tail_ground_token.unsqueeze(0)
            print(
                f"has nan in probs {tail_ground_prob}, and picked token is {tail_ground_token}")
        else:
            tail_ground_token = torch.multinomial(tail_ground_prob, num_samples=1).to(
                device=input_ids.device)
        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        # print(f"shape of input_ids {input_ids.shape}, and  tail_ground_token {tail_ground_token}, tail_ground_token[None] {tail_ground_token[None]}")
        input_ids = torch.cat((input_ids, tail_ground_token[None]), dim=1)

        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            draft_model_accept_indices=draft_keep_indices,
            acceptance_count=depth,
        )

    def verify_single_layer(self,
                            depth: int,
                            idx_group_biases: List,
                            verification_probs_list: List,
                            current_layer_cand_prob_idx,
                            ground_probs):
        """
        idx_group_bias and ground_probs should have the same size - []
        verify cand_probs by levels, but typical acceptance don't need this 
        """
        next_layer_ground_probs = []
        next_layer_idx_group_biases = []
        next_layer_cand_idx = []
        for i in range(len(verification_probs_list)):
            current_ground_prob = verification_probs_list[i]
            idx_group_bias = idx_group_biases[i]
            cand_probs_idx = current_layer_cand_prob_idx[i]

            idx_base = self.cumulative_prod_size[depth] + idx_group_bias
            accept_idx_biases = self.new_verification(
                current_ground_prob,
                self.cand_probs[depth][cand_probs_idx],
                self.unverified_tokens[idx_base: idx_base +
                                       self.tree_config[depth]],
            )
            if len(accept_idx_biases) != 0:
                for accept_idx_bias in accept_idx_biases:
                    global_idx = idx_base + accept_idx_bias
                    next_layer_ground_probs.append(ground_probs[global_idx])
                    # update self.total_path

                    self.update_total_path(idx=self.init_input_length + global_idx,
                                           idx_in_heap=global_idx,
                                           depth=depth)

                    # handle keep_indices after get the full list

                    if depth < self.max_draft_len - 1:
                        cand_probs_idx = idx_group_bias + accept_idx_bias
                        next_layer_cand_idx.append(cand_probs_idx)
                        next_layer_idx_group_biases.append(
                            cand_probs_idx * self.tree_config[depth + 1])
        return next_layer_ground_probs,  next_layer_idx_group_biases, next_layer_cand_idx

    def update_total_path(self,
                          idx,
                          idx_in_heap,
                          depth, ):
        """ 
        1. algorithm that decide which candidate sequence a token idx belong to 
        2. only work with naive k-config not dynamic tree config
        """
        repeat = 1
        last_k = self.prod_size[-1]
        repeat = int(last_k//self.prod_size[depth+1])
        # print(f"what is self_prod size {self.prod_size}")
        k = self.prod_size[depth+1]
        p = self.cumulative_prod_size[depth + 1]
        # print(f'idx is {idx} and idx_in_heap is {idx_in_heap}, k is {k}, p is {p}, depth is {depth}, repeat is {repeat}')
        first_idx = k - (p-idx_in_heap)
        offset = first_idx*repeat
        for _ in range(repeat):
            self.total_path[offset].append(idx)
            offset += 1


class EdgeSideTreeStrategyGeneration:
    def __init__(
        self,
        draft_model,
        use_tree_trunction=False,  # not implemented yet
        tree_config=(2, 2, 1),
        draft_model_temp=1,

    ):
        # assume the model is on device
        self.use_tree_trunction = use_tree_trunction
        # print("type of draft_model is ", type(draft_model))
        self.draft_model = draft_model,
        # print("type of self.draft_model is ", type(self.draft_model))
        self.draft_model_temp = draft_model_temp
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device

        # if passed in tree_config, considered this as traditional tree-attention with fixed configuration
        self.tree_config = tree_config
        prod_size = torch.cumprod(torch.tensor(
            self.tree_config, dtype=torch.int), dim=0)
        prod_size = torch.cat(
            (torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(self.tree_config).to(
            device=self.draft_model_device
        )
        self.draft_model_past_key_values = None

    def update_kv_cache(self, indices):
        if self.draft_model_past_key_values != None:
            for i in range(len(self.draft_model_past_key_values)):
                self.draft_model_past_key_values[i] = (
                    self.draft_model_past_key_values[i][0].index_select(
                        dim=2, index=indices
                    ),
                    self.draft_model_past_key_values[i][1].index_select(
                        dim=2, index=indices
                    ),
                )

    def generate_draft_naive_tree_attn(
        self,
        input_ids,  # already in device from its caller
        past_key_values,  # for kv cache
        draft_model,

    ):
        '''
        1. same as MCSD's draft generation 
        '''
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        init_input_length = input_ids.size(1)
        max_draft_len = len(self.tree_config)
        # print(f"Line 695 what is max_draft_len {max_draft_len}")

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2):]
        else:
            pruned_input_ids = input_ids
        for step in range(max_draft_len):
            step_k = self.tree_config[step]
            if step != 0:
                step_tree_attn_self_mask = self.tree_attn_self_mask[
                    self.cumulative_prod_size[step - 1]: self.cumulative_prod_size[
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
            # print(f"line 682 type of draft_model is {type(self.draft_model)}")
            outputs: BaseModelOutputWithPast = draft_model.model(
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
            logits = draft_model.lm_head(hidden_states)  # seq_len x hidden_dim

            past_key_values = list(outputs.past_key_values)
            self.draft_model_past_key_values = past_key_values

            step_cand_probs = torch.softmax(
                logits / self.draft_model_temp, dim=-1)
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
            tree_config=self.tree_config)

    def dynamic_k(self, logits: torch.Tensor, top_p: float, top_k: int, entropy_threshold: float) -> int:
        """determine k config baesd on current output logits from the model

        Args:
            logits (torch.Tensor, shape: (batch_size, vocab_size)): 
                generated probability distributino over all tokens based on context

        Returns:
            int: k config for next level
        """
        # perform entropy
        log_tolerance = 1e-9
        logits_prob = F.softmax(logits, dim=-1)
        entropy = -(logits_prob * torch.log(logits_prob + log_tolerance)).sum(dim=-1)
        if entropy.item() > entropy_threshold:
            return 0

        # perform top p 
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        print(f"sorted indices to remove {sorted_indices_to_remove}")

        # shift indices to next position
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        print(f"sorted indices to remove after {sorted_indices_to_remove}")

        sorted_logits[sorted_indices_to_remove] = -float('Inf')
        #revert back the logit indeces
        filtered_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
        print("before: ", filtered_logits)
        
        # perform top k
        if top_k < filtered_logits.size(-1):
            _, sorted_indices = torch.sort(filtered_logits, descending=True)
            indices_to_remove = sorted_indices[-1, top_k:]
            print(f"indices_to_remove: {indices_to_remove}")
            filtered_logits[-1, indices_to_remove] = -float("Inf")
        
        print("after: ", filtered_logits)

        indices = filtered_logits != -float('Inf')
        return int(torch.sum(indices).item()), torch.where(filtered_logits != -float('Inf'))[1].tolist()
    
    def generate_draft_dynamic_tree_attn(
        self,
        input_ids,
        past_key_values,
        draft_model,
        top_p=0.9,
        top_k=50,
        entropy_threshold=2.5,
    ):
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        init_input_length = input_ids.size(1)
        max_draft_len = 10  # Set a maximum depth for the dynamic tree
        self.tree_config = []  # Initialize an empty list to store the dynamic tree configuration

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2):]
        else:
            pruned_input_ids = input_ids

        for step in range(max_draft_len):
            outputs: BaseModelOutputWithPast = draft_model.model(
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
            logits = draft_model.lm_head(hidden_states)  # seq_len x hidden_dim

            past_key_values = list(outputs.past_key_values)
            self.draft_model_past_key_values = past_key_values

            step_k, cand_token_indices = self.dynamic_k(logits[0], top_p, top_k, entropy_threshold)

            if step_k == 0:
                break

            self.tree_config.append(step_k)
            cand_tokens = torch.tensor(cand_token_indices, device=self.draft_model_device).unsqueeze(0)

            step_cand_probs = F.softmax(logits / self.draft_model_temp, dim=-1)
            cand_probs.append(step_cand_probs)
            pruned_input_ids = cand_tokens
            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)

            # Generate the tree attention mask for the current step
            step_tree_attn_self_mask = self.get_tree_attn_from_dynamic_k(self.tree_config).to(
                device=self.draft_model_device
            )
            context_attn_mask = torch.ones(
                (step_tree_attn_self_mask.size(0), init_input_length), dtype=torch.bool
            ).to(self.tree_attn_self_mask)
            step_tree_attn_mask = torch.cat(
                (context_attn_mask, step_tree_attn_self_mask), dim=1
            )

            position_ids = torch.arange(
                init_input_length,
                init_input_length + step_tree_attn_self_mask.size(0),
                dtype=torch.long,
                device=self.draft_model_device,
            ).unsqueeze(0)

        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
            tree_config=tuple(self.tree_config),
        )
    
    def get_tree_attn_from_dynamic_k(self, k_config): # could be like ((2), (1,2), (3,0,1))
        token_size = 0
        level_threshold =[]
        for level in k_config:
            token_size += sum(level)
            level_threshold.append(token_size)
        attn_mask = torch.zeros((token_size, token_size), dtype=torch.bool)
        attn_mask = attn_mask.diagonal_scatter(torch.ones(token_size))
        idx_queue = [(0,None,idx) for idx in list(range(k_config[0][0]))]
        while len(idx_queue) !=0:
            depth, parent, idx = idx_queue.pop(0)
            # print(f"{depth}, {parent}, {idx}")
            if parent is not None:
                # update the child
                attn_mask[idx, : parent + 1] = attn_mask[parent, : parent + 1]
            # deepest depth processed is len(k-config)-1
            if depth < len(k_config)-1:
                if depth ==0:
                    idx_base = 0
                else:
                    idx_base = level_threshold[depth-1]
                child_idx_base = level_threshold[depth]
                child_config = k_config[depth+1]
                distance_from_base = idx - idx_base
                # if distance is zero, means the first k in child config
                node_k = child_config[distance_from_base]
                for child_idx_bais in range(node_k):
                    
                    level_child_location = sum(k_config[depth+1][:distance_from_base]) # no need for plus one here, because distance from base start from 1
                    real_child_idx = level_child_location + child_idx_base + child_idx_bais
                    idx_queue.append((depth+1,idx,real_child_idx))
        return attn_mask