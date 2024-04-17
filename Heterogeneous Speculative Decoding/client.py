
from transformers import AutoTokenizer, AutoModelForCausalLM
from hetero_speculative_decoding import hetero_speculative_decoding

if __name__ == '__main__':
    draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="auto", trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
    client = hetero_speculative_decoding()
    SERVER_IP = '192.168.0.132'
    client_id = input("Please input a valid client id: ")  # Assign a unique client ID for each client
    
    input_ids = draft_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    top_k = 20
    top_p = 0.9
    output = client.edge_speculative_decoding(
        input_ids=input_ids,
        draft_model=draft_model,
        server_ip=SERVER_IP,
        max_len=128,
        gamma=4,
        client_id=client_id
    )
    print(f'total time on communication: {client.time_spend_sending_message()}')
    print(f'total time on target model forward: {client.time_spend_on_target_model_forward()}')
    print(f'total time on draft model generation: {client.time_spend_on_draft_model_generation()}')
    print(f'output is {draft_tokenizer.batch_decode(output)}')