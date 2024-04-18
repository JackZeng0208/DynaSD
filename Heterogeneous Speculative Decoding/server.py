import zmq
from hetero_speculative_decoding import hetero_speculative_decoding
from transformers import AutoModelForCausalLM
import torch

if __name__ == "__main__":
    server = hetero_speculative_decoding()
    target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, trust_remote_code=True)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1919")
    print("Server is running...")
    server.server_speculative_decoding(socket, target_model)