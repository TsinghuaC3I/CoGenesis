#!/usr/bin/env python
import os
import json
import srsly
import requests
from copy import deepcopy
from tqdm import tqdm
from fire import Fire
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from tqdm import trange
import random
random.seed(42)
from custom_sampling import StoppingCriteriaList, KeyWordsCriteria, top_k_top_p_filtering
from run_fuse_logits import ProbFusionModel

from ..conv_templates import QwenTemplate
from ..prompts import system_prompt_with_context_ctx, system_prompt_with_context_per, system_prompt_wo_context, ctx_few_shot, paper_few_shot

LLM_API_URL=os.environ.get("LLM_API_URL", "http://192.168.1.8:18000/generate")

task2data = {
    "ctx": "../../data/ablation/ctx_test.json",
    "paper": "../../data/ablation/paper_test.json",
}

task2few_shot = {
    "ctx": ctx_few_shot,
    "paper": paper_few_shot
}

def create_prompt(task, mode, sample, split=True):
    few_shot = task2few_shot[task]
    value = sample["conversations"][0]["value"]
    value = value if task == "ctx" else value.replace("Title: ", "")
    if mode == "with":
        if task == "ctx":
            prompt = system_prompt_with_context_ctx + "\n\n" + few_shot["w_context"].format(
                profile=sample["additional_profile"],
                history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                task=value
            )
        elif task in ["email", "paper"]:
            if task == "email":
                sample['profile'] = sample['profile'][:3]
            prompt = system_prompt_with_context_per + "\n\n" + few_shot["w_context"].format(
                examples="\n\n".join([f"Title: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                task=value
            )
    elif mode == "without":
        if task == "ctx" and split:
            value = value.split(": ")[0].strip()
        prompt = system_prompt_wo_context + "\n\n" + few_shot["wo_context"].format(
                task=value
            )
    return prompt

class LLMNextTokenLogits:
    def __init__(self, temperature=0.7, top_p=0.9, logprobs=10, vocab_size=152064, url=LLM_API_URL, fusion_model_path=None, dtype=torch.float) -> None:
        self.data = {
            "prompt_token_ids": None,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 1,
            "logprobs": logprobs
        }
        self.vocab_size = vocab_size
        self.empty_logits = torch.full((1, self.vocab_size), 0)
        # self.empty_logits = torch.zeros(self.vocab_size)
        self.url = url

        if fusion_model_path is not None:
            self.fusion_model = ProbFusionModel(vocab_size)
            self.fusion_model.load_state_dict(torch.load(fusion_model_path))
            self.fusion_model.to("cuda").to(dtype).eval()

    def next_token_logits(self, token_ids):
        self.data["prompt_token_ids"] = token_ids.tolist()[0]
        res = requests.post(self.url, json=self.data).json()
        # token_ids = list(res["logprobs"][0][0].keys())
        # logprob = res["logprobs"][0][0].values()
        logits = deepcopy(self.empty_logits)
        for token_id, logprob in res["logprobs"][0][0].items():
            logits[0][int(token_id)] = logprob if logprob is not None else 0
        # print("LLM:", res["text"][0], res["logprobs"][0][0])
        return logits

    def fuse_logits(self, logits1, logits2):
        return self.fusion_model(logits1, logits2)

def fusion_func_mean(base_logits, large_logits):
    mask_value = -1e9
    base_logits = torch.where(base_logits == float('-inf'), torch.full_like(base_logits, mask_value), base_logits)
    large_logits = torch.where(large_logits == float('-inf'), torch.full_like(large_logits, mask_value), large_logits)
    return (base_logits + large_logits) / 2
    
def fusion_func_max(base_logits, large_logits):
    return torch.max(base_logits, large_logits)


class FusionStrategy:
    def __init__(self, P=5, K=10, T=3) -> None:
        self.P = P
        self.K = K
        self.T = T
        
    def __call__(self, step):
        """
        Determines if the model should update at the given step.
        
        Parameters:
        - step: The current step number.
        - K: The initial step offset.
        - T: The number of steps to update consecutively after K.
        
        Returns:
        - True if the model should update on the given step, False otherwise.
        """
        # The model updates if the step is in the range [K+1, K+T]
        if step < self.P:
            return True
        # Calculate the phase within the current cycle
        phase = (step - self.K - 1) % (self.K + self.T)
        # The model updates if the phase is in the range [0, T-1]
        return 0 <= phase < self.T

def co_generate(base_model, large_model, tokenizer, input_ids, input_ids_wo, max_new_tokens, temperature, do_sample, top_p, fusion_strategy, **kwargs):
    kwargs["use_cache"] = True
    base_kwargs = kwargs.copy()
    
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    stop_words_ids = base_kwargs.pop("stop_words_ids", None)
    generation_config = base_kwargs.pop("generation_config", base_model.generation_config)
    if stop_words_ids is not None:
        stop_words_ids.append([generation_config.eos_token_id])
        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_words_ids)]) if stop_words_ids else None
        pass
    else:
        stopping_criteria = None

    step_logs = []
    for step in trange(max_new_tokens):
        # prepare model inputs with past_key_values and attention_mask
        base_inputs = base_model.prepare_inputs_for_generation(input_ids, **base_kwargs)
        base_outputs = base_model(
            **base_inputs, return_dict=True
        )
        base_next_token_logits = base_outputs.logits[:, -1, :]
        step_log = {}

        if fusion_strategy(step):
            top_values, topk_indices = torch.topk(base_next_token_logits, 10, dim=1)
            # print("SLM:", {i:v for i, v in zip(topk_indices[0].tolist(), top_values[0].tolist())})
            step_log["slm"] = {i:v for i, v in zip(topk_indices[0].tolist(), top_values[0].tolist())}
            mask = torch.zeros_like(base_next_token_logits, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            base_next_token_logits[~mask] = 0

            large_next_token_logits = large_model.next_token_logits(input_ids).to(base_next_token_logits.device)
            values, indices = torch.topk(large_next_token_logits, 10, dim=-1)
            step_log["llm"] = {i:v for i, v in zip(indices[0].tolist(), values[0].tolist())}
            
            next_token_logits, weights = large_model.fuse_logits(base_next_token_logits, large_next_token_logits)
            step_log["weights"] = weights[0].tolist()
            
            values, indices = torch.topk(next_token_logits, 10, dim=-1)
            step_log["cogen"] = {i:v for i, v in zip(indices[0].tolist(), values[0].tolist())}
        else:
            next_token_logits = base_next_token_logits

        # decode
        if do_sample:
            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            probs = F.softmax(next_token_logits, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids_wo = torch.cat([input_ids_wo, next_tokens[:, None]], dim=-1)
        
        step_log["token"] = tokenizer.decode(next_tokens, skip_special_tokens=False)
        step_logs.append(step_log)
        
        # update kwargs
        base_kwargs = base_model._update_model_kwargs_for_generation(base_outputs, base_kwargs)

        if stopping_criteria and stopping_criteria(input_ids, None):
            break
        
        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            break

    return input_ids, step_logs

def main(task, small_model_path, small_model_name="Qwen1.5-1.8B-Chat", large_model_name="Qwen-72B-Chat", P=1, K=1, T=1):
    fusion_strategy_args = {
        "P": P,
        "K": K,
        "T": T
    }
    tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
    if "Qwen1.5" in small_model_name:
        tokenizer.eos_token_id = tokenizer("<|endoftext|>").input_ids[0]
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(small_model_path, device_map="auto", trust_remote_code=True).eval()
    
    large_model = LLMNextTokenLogits(vocab_size=base_model.config.vocab_size, fusion_model_path=f"outputs/{task}/{small_model_name}_{large_model_name}_best.pt", dtype=base_model.dtype)
    
    data = srsly.read_json(task2data[task])[:50]
    template = QwenTemplate()
    stop_words_ids = [[tid] for tid in template.get_stop_token_ids()]

    fusion_strategy = FusionStrategy(**fusion_strategy_args)
    
    logs = []
    for sample in tqdm(data, desc="Generating"):
        prompt_w = create_prompt(task, "with", sample)
        prompt_wo = create_prompt(task, "without", sample)

        prompt_w = template(prompt_w)
        prompt_wo = template(prompt_wo)
        
        inputs_w = tokenizer(
            prompt_w,
            return_tensors='pt'
        )["input_ids"]

        inputs_wo = tokenizer(
            prompt_wo,
            return_tensors='pt'
        )["input_ids"]
        
        with torch.no_grad():
            co_output_ids, log = co_generate(base_model=base_model,
                                large_model=large_model,
                                tokenizer=tokenizer,
                                input_ids=inputs_w.to(base_model.device), 
                                input_ids_wo=inputs_wo.to(base_model.device), 
                                max_new_tokens=1024, 
                                temperature=0.7, 
                                do_sample=True,
                                top_p=0.9,
                                fusion_strategy=fusion_strategy,
                                stop_words_ids=stop_words_ids)
            model_output = tokenizer.decode(co_output_ids[0], skip_special_tokens=False)

            sample["model_output"] = model_output.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()

            logs.append({
                "prompt": prompt_wo,
                "step_logs": log,
                "tid": sample["tid"],
            })
    
    suffix = "logits_fit" + "_" + "_".join([f"{k}{v}" for k, v in fusion_strategy_args.items()])
    srsly.write_json(f"outputs/results/{task}_{small_model_name}-{large_model_name}_{suffix}.json", data)
    json.dump(logs, open(f"outputs/results/{task}_{small_model_name}-{large_model_name}_log_{suffix}.json", "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    Fire(main)