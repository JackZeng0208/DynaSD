from transformers import AutoTokenizer, AutoModelForCausalLM
from hetero_speculative_decoding import hetero_speculative_decoding
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Heterogeneous Speculative Decoding Client)")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Draft model name")
    parser.add_argument("--dataset", type=str, default="mandarjoshi/trivia_qa", help="Huggingface dataset name (ex: mandarjoshi/trivia_qa)")
    parser.add_argument("--range", nargs=2, type=int, default=[0, 1000], help="Range of dataset to evaluate")
    parser.add_argument("--server_ip", type=str, default="192.168.0.132")
    parser.add_argument("--port", type=str, default="1919", help="Server port number")
    parser.add_argument("--client_id", type=str, default=os.getlogin(), help="Client ID")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--input_text", type=str,
                        default="Please write an introduction about UC Irvine:")
    args = parser.parse_args()
    draft_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
    client = hetero_speculative_decoding()
    input_ids = draft_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    top_k = 20
    top_p = 0.9
    output = client.edge_speculative_decoding(
        input_ids=input_ids,
        draft_model=draft_model,
        server_ip=args.server_ip,
        port=args.port,
        max_len=128,
        gamma=4,
        client_id=args.client_id
    )
    print(f'total time on communication: {client.get_time_spend_sending_message()}')
    print(f'total time on target model forward: {client.get_time_spend_on_target_model_forward()}')
    print(f'total time on draft model generation: {client.get_time_spend_on_draft_model_generation()}')
    print(f'output is {draft_tokenizer.batch_decode(output)}')