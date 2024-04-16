import torch
import torch.distributed as dist 
import torch
import threading
from heterogeneous_utils import norm_logits
import time
import zmq

tensor_lock = threading.Lock()
draft_tokens = None

from transformers import AutoTokenizer, AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto", trust_remote_code=True)
target_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
target_model.to('cuda:0')

@torch.no_grad()
def server_speculative_sampling_without_kvcache(draft_tokens: torch.Tensor, 
                        model: torch.nn.Module, 
                        temperature: float = 1, top_k: int = 0, 
                        top_p: float = 0, verbose: bool = False, 
                        random_seed: int = 1234) -> list:
    target_model_history = model(draft_tokens).logits
    for i in range(target_model_history.shape[-2]):
        target_model_history[:,i,:] = norm_logits(target_model_history[:,i,:], temperature, top_k, top_p)
    return target_model_history

def handle_request(socket):
    global draft_tokens

    while True:
        message = socket.recv_pyobj()
        if message['type'] == 'send_tensor':
            received_draft_tokens = message['draft_tokens']
            with tensor_lock:
                draft_tokens = received_draft_tokens
            socket.send_pyobj({'message': 'server received tokens'})
        elif message['type'] == 'get_tensor':
            with tensor_lock:
                if draft_tokens is not None:
                    draft_tokens = draft_tokens.to("cuda:0")
                    target_forward_time = time.time()
                    target_model_history_tensor = server_speculative_sampling_without_kvcache(
                        draft_tokens=draft_tokens,
                        model=target_model
                    )
                    finish_target_forward_time = time.time()

                    draft_tokens = None
                    response = {
                        'target_prob_hist': target_model_history_tensor,
                        'target_model_generation_time': finish_target_forward_time - target_forward_time,
                    }
                else:
                    response = {'target_prob_hist': None}
            socket.send_pyobj(response)

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1919")

    print("Server is running...")

    handle_request(socket)
