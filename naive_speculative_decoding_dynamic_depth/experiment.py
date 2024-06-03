from speculative_sampling import speculative_sampling

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
import time


model_name = {'apple270m':{'name':'apple/OpenELM-270M-Instruct','datatype':torch.bfloat16}, 
              'apple3b':{'name':'apple/OpenELM-3B-Instruct','datatype':torch.bfloat16},
              'llamatiny':{'name':'TinyLlama/TinyLlama-1.1B-Chat-v1.0','datatype':torch.bfloat16},
              'llama68m':{'name':'JackFram/llama-68m','datatype':torch.bfloat16},
              'llama160m':{'name':'JackFram/llama-160m','datatype':torch.bfloat16},
              'llama7b':{'name':'meta-llama/Llama-2-7b-chat-hf','datatype':torch.bfloat16}}

sentence_list = ["Write a story set in a post-apocalyptic world where humans are struggling to survive.",
"Describe a magical forest where every tree has a unique power.",
"Imagine a future where artificial intelligence has surpassed human intelligence.",
"Create a dialogue between a time traveler and a historical figure.",
"Describe a society where emotions are considered a form of currency.",
"Write a poem about the beauty and power of the ocean.",
"Invent a new planet with its own ecosystem and inhabitants.",
"Describe a day in the life of a character who can teleport at will.",
"Write a short story about a secret society hiding in plain sight.",
"Imagine a world where dreams are tangible and can be bought and sold.",
"Once upon a time"]

target_model_name = model_name['llama7b']['name']
draft_model_name =model_name['llama68m']['name']
print(model_name['llama68m']['name'] =='JackFram/llama-68m')
draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float32,
        # load_in_8bit = True,
        # device_map=0,
        # use_flash_attention_2= False,
    ).to('cuda')

target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        # load_in_8bit = True,
        # device_map=0,
        # use_flash_attention_2= False,
    ).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

max_length = 200


def eval_speed(fn, target_model,draft_model,tokenizer=tokenizer,max_length = max_length,eval_dataset = sentence_list,msg = 'feifeibear'):
    start = time.time()
    eval_stats = {'acceptance_rate':0.0,
             'target_forward_count':0,
             'draft_forward_count':0,
             'draft_generation_time':0,
             'target_foward_time':0}
    output_len = 0
    for s in eval_dataset:
        input_ids = tokenizer(s, return_tensors='pt').input_ids.to('cuda')
        if (target_model != None and fn != None):
            output,stats = fn(prefix=input_ids,gamma = 8,
                                        target_model=target_model,
                                        approx_model = draft_model,
                                        max_len=max_length)
            eval_stats['acceptance_rate'] += stats['acceptance_rate']
            eval_stats['draft_forward_count']+= stats['draft_forward_count']
            eval_stats['draft_generation_time']+= stats['draft_generation_time']
            eval_stats['target_forward_count']+= stats['target_forward_count']
            eval_stats['target_foward_time']+= stats['target_foward_time']
            
        elif target_model == None and fn != None:
            # output = fn(input_ids,assistant_model =draft_model,max_new_tokens =max_length,do_sample=True,
            #         temperature=1,
            #         top_k=10,
            #         top_p=0.9,)
            output = fn(input_ids,assistant_model =draft_model,max_new_tokens =max_length)
        elif draft_model == None and fn == None:
            output = target_model.generate(input_ids,max_new_tokens =max_length,do_sample=True,
                    temperature=1,
                    top_k=10,
                    top_p=0.9)
        print(f'check input_ids shape {input_ids.shape}')
        output_len += output.shape[1] - input_ids.shape[1]
        print(f'output_len.shape is {output.shape}')
        print(f'output: {tokenizer.batch_decode(output)}')
        print()
    end = time.time()
    eval_stats['acceptance_rate'] = eval_stats['acceptance_rate']/len(eval_dataset)
    print(f"total generation time is {end-start}s")
    print(f"speed for {msg}'s speculative decoding is {(output_len)/(end-start)}t/s")
    for k,v in eval_stats.items():
        print(f"{k} : {v}")
eval_speed(speculative_sampling,target_model,draft_model)
# eval_speed(target_model.generate,None,draft_model,msg="huggingface speculative decoding")
# eval_speed(None,target_model,draft_model=None,msg="no speculative decoding")

