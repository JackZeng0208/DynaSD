import torch
import time
from utils import sample, norm_logits, max_fn, KVCacheModel
import zmq

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
            print(f"this turn accepted rate {accepted_count/gamma}")
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
        while True:
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