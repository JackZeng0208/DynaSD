import zmq
from single_cand_speculative_decoding import hetero_speculative_decoding
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Server)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf", help="Target model name")
    parser.add_argument("--port", type=str, default="1919")
    args = parser.parse_args()
    server = hetero_speculative_decoding()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )
    # print(f"what is target device {target_model.device}")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    # # Set send buffer size to 1 MB
    socket.setsockopt(zmq.SNDBUF, 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
    socket.bind(f"tcp://*:{args.port}")
    print("Server is running...")
    server.server_speculative_decoding(
        server_socket= socket,
        target_model=target_model
    )