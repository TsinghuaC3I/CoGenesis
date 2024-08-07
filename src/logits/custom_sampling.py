#!/usr/bin/env python
import torch
from torch.nn import functional as F
from tqdm import trange
from transformers.generation.utils import (
    top_k_top_p_filtering,
    StoppingCriteria,
    StoppingCriteriaList,
)
from fastchat.conversation import Conversation, SeparatorStyle

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

def generate(model, tokenizer, input_ids, max_new_tokens, temperature, do_sample, top_p, top_k=0, use_tqdm=False, **base_kwargs):
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    stop_words_ids = base_kwargs.pop("stop_words_ids", None)
    generation_config = base_kwargs.pop("generation_config", model.generation_config)
    if stop_words_ids is not None:
        stop_words_ids.append([generation_config.eos_token_id])
        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_words_ids)]) if stop_words_ids else None
        pass
    else:
        stopping_criteria = None

    for _ in trange(max_new_tokens, disable=not use_tqdm):
        # prepare model inputs with past_key_values and attention_mask
        base_inputs = model.prepare_inputs_for_generation(input_ids, **base_kwargs)
        base_outputs = model(
            **base_inputs, return_dict=True
        )
        base_next_token_logits = base_outputs.logits[:, -1, :]

        # TODO: process logits
        next_token_logits = base_next_token_logits

        # decode
        if do_sample:
            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p, top_k=top_k)

            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # update kwargs
        base_kwargs = model._update_model_kwargs_for_generation(base_outputs, base_kwargs)
        
        if stopping_criteria and stopping_criteria(input_ids, None):
            break
        
        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            break

    print("next_token_logits", next_token_logits.shape)
    return input_ids

def get_prompt(prompt):
    standard_conv = Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
    
    conv = standard_conv.copy()
    conv.set_system_message("You will write beautiful compliments according to needs")
    conv.append_message("<|im_start|>user", prompt)
    conv.append_message("<|im_start|>assistant", None)
    return conv.get_prompt()
