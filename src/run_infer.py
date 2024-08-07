#!/usr/bin/env python

import os
import sys
import srsly
import fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path,
        test_path, 
        data_dir,
        output_dir=None,
        output=None, 
        num_beams=4, 
        do_sample=False, 
        top_p=0.7, 
        temperature=0.9):
    system_prompt = "You are now a helpful personal AI assistant. You should emulate the author's style and tone based on provided history content. Your responses should be detailed and informative, using the personal information reasonably in the user's profile. Aim for insightful and high-quality solutions that make users satisfied."

    test_file = f"{data_dir}/{test_path}"
    print("test_file: ", test_file)
    data = srsly.read_json(test_file)[:100]

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    for sample in tqdm(data):
        user_input = sample["conversations"][0]["value"]
        response, _ = model.chat(tokenizer, user_input, history=None, system=system_prompt, max_new_tokens=4096,
                                 num_beams=num_beams,
                                 do_sample=do_sample, top_p=top_p, temperature=temperature)
        sample["model_output"] = response

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(model_path if output_dir is None else output_dir, output if output else "generations.json")

    srsly.write_json(save_path, data)


if __name__ == "__main__":
    fire.Fire(main)