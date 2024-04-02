import torch
import time
from heterogeneous_utils import KVCacheModel, sample, max_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
import zmq

class stats:
    def __init__(self):
        self.time_spend_sending_message = 0
        self.time_spend_on_draft_model_generation = 0
        self.time_spend_on_target_model_forward = 0

heterogeneous_stats = stats()

def edge_speculative_sampling(prefix: torch.Tensor,
                              approx_model: torch.nn.Module,
                              SERVER_IP: str,
                              max_len: int, gamma: int = 4,
                              temperature: float = 1, top_k: int = 0, top_p: float = 0,
                              verbose: bool = False, random_seed: int = 1234) -> torch.Tensor:
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p).to('cuda:0')

    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    prefix = prefix.to('cuda:0')
    start_time = time.time()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{SERVER_IP}:1919")

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]

        draft_generate_start_time = time.time()
        draft_tokens = approx_model_cache.generate(prefix, gamma)
        draft_generate_end_time = time.time()
        heterogeneous_stats.time_spend_on_draft_model_generation += draft_generate_end_time - draft_generate_start_time


        send_tensor_start_time = time.time()
        socket.send_pyobj({'type': 'send_tensor', 'draft_tokens': draft_tokens})
        response = socket.recv_pyobj()
        print('send from edge to server', response)

        socket.send_pyobj({'type': 'get_tensor'})
        target_model_mesg_dict = socket.recv_pyobj()
        send_tensor_end_time = time.time()

        target_model_history = target_model_mesg_dict['target_prob_hist']
        target_model_generation_time = target_model_mesg_dict['target_model_generation_time']
        total_time_in_server = target_model_generation_time
        heterogeneous_stats.time_spend_sending_message += send_tensor_end_time - send_tensor_start_time - total_time_in_server
        heterogeneous_stats.time_spend_on_target_model_forward += target_model_generation_time

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
        prefix = draft_tokens[:, :n + 1]
        approx_model_cache.rollback(n + 1)
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"

        if n < prefix_len + gamma - 1:
            t = sample(max_fn(target_model_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            resample_count += 1
        else:
            assert n == target_model_history.shape[1] - 1
            t = sample(target_model_history[:, -1, :])
            target_sample_count += 1

        prefix = prefix.to("cuda:0")
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    end_time = time.time()
    print(f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
    print(f"Token Generation Speed (with speculative decoding): {max_len / (end_time - start_time)} tokens/s")
    print(f"Acceptance Rate: {accepted_count / max_len}")
    return prefix

if __name__ == '__main__':
    SERVER_IP = '192.168.0.132'
    approx_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto", trust_remote_code=True)
    approx_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
    input_ids = approx_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    top_k = 20
    top_p = 0.9
    output = edge_speculative_sampling(
        prefix=input_ids,
        approx_model=approx_model,
        SERVER_IP=SERVER_IP,
        max_len=100,
        gamma=4,
    )
    print(f'total time on communication: {heterogeneous_stats.time_spend_sending_message}')
    print(f'total time on target model forward: {heterogeneous_stats.time_spend_on_target_model_forward}')
    print(f'total time on draft model generation: {heterogeneous_stats.time_spend_on_draft_model_generation}')
    print(f'output is {approx_tokenizer.batch_decode(output)}')