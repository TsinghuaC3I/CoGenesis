#!/usr/bin/env python
import srsly
import random
random.seed(42)
import os
import torch
from tqdm import tqdm, trange
from fire import Fire
from colorama import Fore, Style
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


from preprocess import preprocess_qwen, preprocess_tinyllama
from ..prompts import system_prompt_with_context_ctx, system_prompt_with_context_per, system_prompt_wo_context, ctx_few_shot, paper_few_shot

task2data = {
    "ctx": "../../data/ablation/ctx_test.json",
    "paper": "../../data/ablation/ctx_test.json"
}

task2few_shot = {
    "ctx": ctx_few_shot,
    "paper": paper_few_shot
}

def create_prompt(task, mode, sample, return_system_prompt=False):
    few_shot = task2few_shot[task]
    value = sample["conversations"][0]["value"]
    value = value if task == "ctx" else value.replace("Title: ", "")
    if mode == "with":
        if task == "ctx":
            system_prompt = system_prompt_with_context_ctx
            prompt = few_shot["w_context"].format(
                profile=sample["additional_profile"],
                history="\n\n".join([f"Task: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                task=value
            )
        elif task in ["email", "paper"]:
            if task == "email":
                sample['profile'] = sample['profile'][:3]
            system_prompt = system_prompt_with_context_per
            prompt = few_shot["w_context"].format(
                examples="\n\n".join([f"Title: {p['title']}\nContent: {str(p['text'])}" for p in sample['profile']]),
                task=value
            )
    elif mode == "without":
        system_prompt = system_prompt_wo_context
        prompt = few_shot["wo_context"].format(
                task=value
            )
    if return_system_prompt:
        return prompt, system_prompt
    else:
        return system_prompt +"\n\n" + prompt, None

def run_request_model_vllm(model_name_or_path, model_name, mode, tp_size, bsz, gmu=0.8, topK=10):

    print(Fore.GREEN + f"Model: {model_name}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Model: {model_name_or_path}" + Style.RESET_ALL)
    
    if "qwen" in model_name.lower():
        preprocess = preprocess_qwen
    elif "tinyllama"in model_name.lower():
        preprocess = preprocess_tinyllama
    elif "llama" in model_name.lower():
        preprocess = None

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(model=model_name_or_path, trust_remote_code=True, gpu_memory_utilization=gmu, tensor_parallel_size=tp_size, dtype=torch.bfloat16)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1, prompt_logprobs=topK)

    for task in ["paper"]:
        for split in ["train", "dev"]:
            print(Fore.GREEN + f"Task: {task}, Split: {split}" + Style.RESET_ALL)
            
            data = srsly.read_json(task2data[task].replace("test", split))

            user_prompts, system_prompts, assistant_prompts = [], [], []
            for sample in tqdm(data, desc="Generating", total=len(data)):
                prompt, system_prompt = create_prompt(task, mode, sample, return_system_prompt=True)

                assistant = sample["conversations"][1]["value"]
                
                user_prompts.append(prompt)
                system_prompts.append(system_prompt)
                assistant_prompts.append(assistant)

            inputs = preprocess(user_prompts, assistant_prompts, system_prompts, tokenizer, is_padding=False)

            prompt_logits = []
            for i in trange(0, len(inputs["input_ids"]), bsz):
                outputs = llm.generate(prompt_token_ids=inputs["input_ids"][i:i+bsz], sampling_params=sampling_params, use_tqdm=False)
                prompt_logits.extend([output.prompt_logprobs for output in outputs])
                
            # outputs = llm.generate(prompt_token_ids=inputs["input_ids"], sampling_params=sampling_params)
            # prompt_logits = [output.logprobs for output in outputs]

            inputs["logits"] = prompt_logits
            
            if os.path.exists("outputs") == False:
                os.mkdir("outputs")
            torch.save(inputs, f"outputs/{task}_{split}_{mode}_{model_name}.pt")

if __name__ == "__main__":
    # https://github.com/vllm-project/vllm/issues/565#issuecomment-1725174811
    Fire(run_request_model_vllm)
