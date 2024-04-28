import zmq
from speculative_decoding_with_tree import*
from model.llama_tree_attn import LlamaForCausalLM
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Server)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf", help="Target model name")
    parser.add_argument("--port", type=str, default="1919")
    args = parser.parse_args()
    server = hetero_speculative_decoding()
    target_model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        # load_in_8bit = True,
        device_map=0,
        use_flash_attention_2= False,
    )
    print(f"what is target device {target_model.device}")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # # Set send buffer size to 1 MB
    socket.setsockopt(zmq.SNDBUF, 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
    socket.bind(f"tcp://*:{args.port}")
    print("Server is running...")
    server.server_tree_attn_speculative_decoding(
        socket= socket,
        target_model=target_model
    )