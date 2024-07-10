from speculative_sampling import* 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
import time
import json


def load_dataset(filenames):
    dataset = []
    for filename in filenames:
        with open(filename, 'r') as f:
            data = json.load(f)
            for item in data:
                if 'question' in item:
                    dataset.append(item['question'])
                elif 'goal' in item:
                    dataset.append(item['goal'])
    return dataset

filenames = ['/home/iasl-transformers/UCI-IASL-Transformer/evaluation/piqa.json']
dataset = load_dataset(filenames)
numbet_to_test = 20
def eval_speed(fn, target_model, draft_model, tokenizer, max_length, eval_dataset=dataset, option='default'):
    """
        option = "default" -> regular inference using model.generate
        option = "sd" -> hand-written speculative decoding
        option = "transformer_sd" -> huggingface version of speculative decoding
        option = "our" -> our version
    """
    file_path = "/home/iasl-transformers/UCI-IASL-Transformer/naive_speculative_decoding_dynamic_depth/training_data_piqa.csv"
    start = time.time()
    eval_stats = {'acceptance_rate': 0.0,
                  'target_forward_count': 0,
                  'draft_forward_count': 0,
                  'draft_generation_time': 0,
                  'target_foward_time': 0}
    output_len = 0
    for s in tqdm.tqdm(eval_dataset[:numbet_to_test]):
        input_ids = tokenizer(s, return_tensors='pt').input_ids.to('cuda')
        # input_ids = tokenizer(s, return_tensors='pt').input_ids
        if option == "sd":
            if fn == speculative_sampling:
                output, stats = fn(prefix=input_ids, gamma=8,
                                   target_model=target_model,
                                   approx_model=draft_model,
                                   max_len=max_length,
                                   collect_data=False,
                                   file_path=file_path)
            else:
                output, stats = fn(prefix=input_ids, gamma=8,
                                target_model=target_model,
                                approx_model=draft_model,
                                max_len=max_length)
                print(f"Output: {tokenizer.batch_decode(output)}")
            eval_stats['acceptance_rate'] += stats['acceptance_rate']
            eval_stats['draft_forward_count'] += stats['draft_forward_count']
            eval_stats['draft_generation_time'] += stats['draft_generation_time']
            eval_stats['target_forward_count'] += stats['target_forward_count']
            eval_stats['target_foward_time'] += stats['target_foward_time']
            
        elif option == "transformer_sd":
            output = fn(input_ids, assistant_model=draft_model,
                        max_new_tokens=max_length)
        elif option == "default":
            output = target_model.generate(input_ids, max_new_tokens=max_length, do_sample=True,
                                           temperature=1,
                                           top_k=10,
                                           top_p=0.9)
        # print(f'check input_ids shape {input_ids.shape}')
        output_len += output.shape[1] - input_ids.shape[1]
        # print(f'output_len.shape is {output.shape}')
        # print(f'output: {tokenizer.batch_decode(output)}')
        # print()
    end = time.time()
    # FIXME: the len(eval_dataset) changed
    eval_stats['acceptance_rate'] = eval_stats['acceptance_rate'] / numbet_to_test#len(eval_dataset)
    print(f"total generation time is {end-start}s")
    print(f"speed for {option} is {(output_len)/(end-start)}t/s")
    for k, v in eval_stats.items():
        print(f"{k} : {v}")

model_name = {'apple270m': {'name': 'apple/OpenELM-270M-Instruct', 'datatype': torch.bfloat16},
              'apple3b': {'name': 'apple/OpenELM-3B-Instruct', 'datatype': torch.bfloat16},
              'llamatiny': {'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'datatype': torch.bfloat16},
              'llama68m': {'name': 'JackFram/llama-68m', 'datatype': torch.bfloat16},
              'llama160m': {'name': 'JackFram/llama-160m', 'datatype': torch.bfloat16},
              'llama7b': {'name': 'meta-llama/Llama-2-7b-chat-hf', 'datatype': torch.bfloat16}}

target_model_name = model_name['llama7b']['name']
draft_model_name = model_name['llama68m']['name']
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name,torch_dtype=torch.float32).to('cuda')
target_model = AutoModelForCausalLM.from_pretrained(target_model_name,torch_dtype=torch.bfloat16).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
max_length = 200


# option = "default" -> regular inference using model.generate
# option = "sd" -> hand-written speculative decoding (including our version)
# option = "transformer_sd" -> huggingface version of speculative decoding
option = "sd"
eval_speed(target_first_speculative_decoding, target_model, draft_model, tokenizer=tokenizer, option=option, max_length=max_length)
# eval_speed(speculative_sampling, target_model, draft_model, tokenizer=tokenizer, option=option, max_length=max_length)