from . import mcstrategy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

@dataclass
class DecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using MCSD.
    """

    sequences: torch.LongTensor
    acceptance_count: int = None
    draft_token_count: int = None
    invocation_count: int = None

class SpeculativeGenerator:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        use_origin = False
    ) -> None:
        self.tree_config = k_config
        self.use_origin = use_origin
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        # self.strategy: strategies.Strategy = None

     
        self.strategy = mcstrategy.TreeStrategy(
            draft_model=draft_model,
            target_model=target_model,
            k_config=k_config,
            draft_model_temp=draft_model_temp,
            target_model_temp=target_model_temp,
            replacement=replacement,
            speculative_sampling=speculative_sampling,
        )
        

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        target_model_past_key_values = None
        draft_model_past_key_values = None

        invocation_count = 0
        acceptance_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            draft_output = self.strategy.generate_draft(
                input_ids,
                past_key_values=draft_model_past_key_values,
            )

            draft_model_past_key_values = draft_output.past_key_values

            # #origin: 
            if self.use_origin:
                verification_output = self.strategy.verify(
                    input_ids=draft_output.sequences,
                    target_model_past_key_values=target_model_past_key_values,
                    draft_model_past_key_values=draft_output.past_key_values,
                    cand_probs=draft_output.cand_probs,
                )
            else: 
                verification_output = self.strategy.verify_longest_candidate_hetero(
                    input_ids=draft_output.sequences,
                    target_model_past_key_values=target_model_past_key_values,
                    draft_model_past_key_values=draft_output.past_key_values,
                    cand_probs=draft_output.cand_probs,
                    tree_config=self.tree_config
                )

            input_ids = verification_output.sequences

            draft_model_past_key_values = (
                verification_output.draft_model_past_key_values
            )
            target_model_past_key_values = (
                verification_output.target_model_past_key_values
            )

            invocation_count += 1
            acceptance_count += verification_output.acceptance_count

            if (
                self.eos_token_id in input_ids[0, -self.strategy.max_draft_len :]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(
            sequences=input_ids,
            acceptance_count=acceptance_count,
            draft_token_count=invocation_count * self.strategy.max_draft_len,
            invocation_count=invocation_count,
        )