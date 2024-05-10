from dynamic_tree_test import (
    ServerSideVerification,
    HeteroSpeculativeDecoding, 
    DecoderOnlyDraftOutput,
    DecoderOnlyVerificationOutput,
    EdgeSideTreeStrategyGeneration
)
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer
)

from typing import List, Dict
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
import torch
import os
import zmq
import argparse
import logging
from tqdm import tqdm


def evaluate(
    inputs: List[Dict[str, str]],
    model_name,
    server_ip,
    port,
    client_id
):
    approx_model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True)
    approx_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    approx_model.to('cuda:0')
    client = HeteroSpeculativeDecoding()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # Set send buffer size to 1 MB
    socket.setsockopt(zmq.SNDBUF, 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
    socket.connect(f"tcp://{server_ip}:{port}")
    total_acceptace_rate = 0
    for example in tqdm(inputs):
        question = example["question"]
        input_str = f"Question: {question}\nAnswer:"
        input_ids = approx_tokenizer.encode(input_str, return_tensors='pt')

        # TODO: compare entropy variantion of each prediction step between draft and target model outputs
        output, acceptance_rate = client.edge_tree_attn_speculative_decoding(
            input_ids=input_ids,
            draft_model=approx_model,
            edge_socket=socket,
            max_len=128,
            client_id=client_id
        )
        total_acceptace_rate += acceptance_rate



if __name__ == "__main__":
    # file_name = "entropy_comparison.txt"
    # file_path = os.path.join(os.getcwd(), file_name)
    # with open(file_path, mode="w") as file:
    #     writer = csv.writer(file)
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Evaluation"
    )
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Draft model name")
    parser.add_argument("--server_ip", type=str, default="192.168.0.132")
    parser.add_argument("--port", type=str, default="1919",
                        help="Server port number")
    parser.add_argument("--client_id", type=str,
                        default=os.getlogin(), help="Client ID")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    args = parser.parse_args()
    
    inputs = [
        {"question": "Tell me about UC Irvine"},
    ]
    
    evaluate(
        inputs,
        model_name=args.model_name,
        server_ip=args.server_ip,
        port=args.port,
        client_id=args.client_id
    )
    
    
    