
# Reference: 
import time 
import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from DynaSD.decision_models import  *
from scipy import stats
import pickle

class TargetInitMultiCandidates:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id,
        config_width = 6,
        config_depth = 10,
        max_new_tokens: int = 200,
        greedy = False, # sampling method greedy or speculative sampling 
        generate_training_data = False, # generating training data to train decision model 
        draft_model_temp: float = 0,
        target_model_temp: float = 0,
    ) -> None:
        """
        so the max_config here could be 5 in width and 10 in depth 
        """
        
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device
        self.target_model_device = (
            target_model.model.get_input_embeddings().weight.device
        )
        # for decision model 
        self.stop_generation = False
        self.continue_depth = 0 
        self.dynamic_target_mask =None


        self.greedy = greedy
        if draft_model_temp >0 or target_model_temp>0 and greedy == True:
            print(f"temperature is non zero, greedy turn to false")
            self.greedy = False
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.max_config = self.generate_fork_config(width=config_width,depth=config_depth)
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_draft_len = len(self.max_config)-1
        self.draft_model_temp = draft_model_temp
        self.target_model_temp = target_model_temp
        self.target_pastjkey_values = None
        self.draft_past_key_values = None

        prod_size = torch.cumprod(torch.tensor(self.max_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(self.max_config).to(
            device=self.draft_model_device
        )

        

        # decision model train data collection 
        self.draft_hidden_states = None # input of decision model 
        self.verification_result = None 
        self.draft_entropy = None
        self.draft_topk_prob = None
        self.generate_training_data = generate_training_data

        # stats collection
        self.stats = {'ground_acceptance_count': 0,
                       "draft_generation_time":0.0, 
                       "verification_time":0.0, 
                       "total_generation_round":0, 
                       "decision_model_time":0.0, 
                       "decision_acceptance_count":0,
                       "total_generated_draft_tokens":0}
        self.max_token_path = []
        for _ in range(config_width):
            self.max_token_path.append([])