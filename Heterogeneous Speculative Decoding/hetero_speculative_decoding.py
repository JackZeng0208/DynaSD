import torch
import threading
import time
import zmq
from transformers import AutoModelForCausalLM
from utils import sample, norm_logits, max_fn, KVCacheModel


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
        self.tensor_lock = threading.Lock()
        self.server_draft_tokens = None
        self.edge_draft_tokens = None

    def get_time_spend_on_draft_model_generation(self):
        return self.time_spend_on_draft_model_generation

    def get_time_spend_on_target_model_forward(self):
        return self.time_spend_on_target_model_forward

    def get_time_spend_sending_message(self):
        return self.time_spend_sending_message

    def edge_speculative_decoding(self,
                                  input_ids: torch.Tensor,
                                  draft_model: torch.nn.Module,
                                  server_ip: str,
                                  max_len: int,
                                  gamma: int = 4,
                                  temperature: float = 1,
                                  top_k: int = 0,
                                  top_p: float = 0,
                                  random_seed: int = 1234,
                                  client_id: str = "") -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): input tensor
            draft_model (torch.nn.Module): draft model for speculative decoding
            server_ip (str): server IP
            max_len (int): maximum length
            gamma (int, optional): gamma. Defaults to 4.
            temperature (float, optional): temperature. Defaults to 1.
            top_k (int, optional): top k. Defaults to 0.
            top_p (float, optional): top p. Defaults to 0.
            random_seed (int, optional): random seed. Defaults to 1234.
            client_id (str, optional): client ID. Defaults to None.
        """
        seq_len = input_ids.shape[1]
        T = seq_len + max_len
        approx_model_cache = KVCacheModel(
            draft_model, temperature, top_k, top_p).to('cuda:0')

        resample_count = 0
        target_sample_count = 0
        accepted_count = 0
        input_ids = input_ids.to('cuda:0')
        start_time = time.time()

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{server_ip}:1919")

        while input_ids.shape[1] < T:
            prefix_len = input_ids.shape[1]

            draft_generate_start_time = time.time()
            self.edge_draft_tokens = approx_model_cache.generate(
                input_ids, gamma)
            draft_generate_end_time = time.time()
            self.time_spend_on_draft_model_generation += draft_generate_end_time - draft_generate_start_time

            send_tensor_start_time = time.time()
            socket.send_pyobj(
                {'type': 'send_tensor', 'draft_tokens': self.edge_draft_tokens, 'client_id': client_id})
            response = socket.recv_pyobj()

            socket.send_pyobj({'type': 'get_tensor', 'client_id': client_id})
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
                j = self.edge_draft_tokens[:, prefix_len + i]

                if r > (target_model_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n = prefix_len + i - 1
                    break
                accepted_count += 1

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            input_ids = self.edge_draft_tokens[:, :n + 1]
            approx_model_cache.rollback(n + 1)
            assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"

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
            print(f"generated tokens numbers {input_ids.shape[-1] - seq_len}, accepted_count {
                  accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        end_time = time.time()
        print(f'total time spend on heterogeneous speculative decoding: {
              end_time - start_time}')
        print(f"Token Generation Speed (with speculative decoding): {
              max_len / (end_time - start_time)} tokens/s")
        print(f"Acceptance Rate: {accepted_count / max_len}")
        approx_model_cache.clear_cache()
        return input_ids

    @torch.no_grad()
    def sampling_without_kvcache(self, draft_tokens: torch.Tensor,
                                                    target_model: torch.nn.Module,
                                                    temperature: float = 1, top_k: int = 0,
                                                    top_p: float = 0,
                                                    random_seed: int = 1234) -> list:
        target_model_history = target_model(draft_tokens).logits
        for i in range(target_model_history.shape[-2]):
            target_model_history[:, i, :] = norm_logits(
                target_model_history[:, i, :], temperature, top_k, top_p)
        return target_model_history

    def server_speculative_decoding(self, socket, target_model: torch.nn.Module):
        while True:
            message = socket.recv_pyobj()
            if message['type'] == 'send_tensor':
                received_draft_tokens = message['draft_tokens']
                with self.tensor_lock:
                    self.server_draft_tokens = received_draft_tokens
                socket.send_pyobj({'message': 'server received tokens'})
            elif message['type'] == 'get_tensor':
                with self.tensor_lock:
                    if self.server_draft_tokens is not None:
                        self.server_draft_tokens = self.server_draft_tokens.to(
                            "cuda:0")
                        target_forward_time = time.time()
                        target_model_history_tensor = self.sampling_without_kvcache(
                            target_model=target_model,
                            draft_tokens=self.server_draft_tokens
                        )
                        finish_target_forward_time = time.time()

                        self.server_draft_tokens = None
                        response = {
                            'target_prob_hist': target_model_history_tensor,
                            'target_model_generation_time': finish_target_forward_time - target_forward_time,
                        }
                    else:
                        response = {'target_prob_hist': None}
                socket.send_pyobj(response)
