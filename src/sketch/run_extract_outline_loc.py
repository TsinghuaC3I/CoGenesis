#!/usr/bin/env python
import os
import srsly
from tqdm import tqdm
from fire import Fire
from colorama import Fore, Style
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from ..conv_templates import MistralTemplate, QwenTemplate, DeepSeekTemplate, StableLMTemplate, TinyLlamaTemplate, H2OTemplate, LlamaTemplate

skeleton_extract_template_ctx = """You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, each skeleton point should be very short with only 3-5 words. Generally, the skeleton should have 8-15 points. You can refer to the following examples:

[Task1]: Develop a Marketing Script for Your Monthly Dinner Party: Create a script that highlights your monthly dinner party as a networking platform.
[Skeleton1]: 1. Warmly lit dining room\n2. Fine china and gourmet dishes\n3. Soft music background\n4. Invitation opening\n5. Guests arriving and networking\n6. Host's welcoming toast\n7. Expertly paired courses and wine\n8. Animated guest discussions\n9. Guest speaker's address\n10. Post-dinner networking lounge\n11. Online community continuation\n12. Next event date highlighted\n13. Closing with logo and contact info

[Task1]: Compose a reflective essay on the evolution of bridge design: Thomas, with his patent in bridge design, can discuss the evolution of bridge engineering, modern challenges, and future perspectives.
[Skeleton1]: 1. Introduction to bridges\n2. Early bridges: materials, principles\n3. Roman arches, concrete use\n4. Industrial Revolution: iron, steel\n5. Brooklyn Bridge: design icon\n6. 20th-century advances: materials, techniques\n7. Modern challenges: sustainability, climate\n8. Future technologies: smart materials, sensors\n9. Ethical considerations, safety\n10. Conclusion: adaptation, advancement

Now, please provide the skeleton for the following question.
{question}
"""

skeleton_extract_template_email = """You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, each skeleton point should be very short with only 3-5 words. Generally, the skeleton should have 8-15 points. You can refer to the following examples:

[Task1]: Compose an email for the subject 'T-Mobile Sidekick debuts, FileMaker launches mobile DB, and more!'
[Skeleton1]: 1. JavaWorld techno-tidbits intro\n2. T-Mobile Sidekick debut\n3. FileMaker mobile DB launch\n4. Palm OS 5 devices release\n5. Mobile security advancements\n6. Newsletter system update\n7. Customer service instructions\n8. JavaWorld team sign-off\n9. Editorial and advertising contacts\n10. Privacy policy reminder\n11. Copyright notice

[Task1]: Compose an email for the subject 'tomcat4, where servlet.jar is set ???'
[Skeleton1]: 1. Tomcat 4 servlet.jar location?\n2. Navigating Tomcat directory.\n3. Specifics for Tomcat 4.\n4. Setting up web application.\n5. Importance of servlets.\n6. Documentation exploration.\n7. Request for expert advice.\n8. Configuration file settings?\n9. Thanks and anticipation.\n10. P.S. Collaboration value.

Now, please provide the skeleton for the following question.
{question}
"""

skeleton_extract_template_paper = """You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question from high-level perspective. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, each skeleton point should be very short with only few words. Generally, the skeleton should have 8-15 points. You can refer to the following examples:

[Task1]: Compose an abstract for the title 'Ensemble of Anchor Adapters for Transfer Learning'
[Skeleton1]: 1. Transfer learning importance\n2. Traditional approaches limitations\n3. Ensemble of Anchor Adapters introduction\n4. Anchor adapters concept\n5. Ensemble strategy for robustness\n6. Hybrid loss function formulation\n7. Experiments on heterogeneous domains\n8. EAA outperforms state-of-the-art\n9. Novel transferability metric introduction\n10. Contribution: ensemble and domain adaptation integration

[Task1]: Compose an abstract for the title 'Variability in software architecture: the road ahead'
[Skeleton1]: 1. Software architecture evolution\n2. VARSA symposium introduction\n3. Previous work foundation\n4. Challenges and opportunities\n5. Keynote speeches, research, collaboration\n6. Capturing and leveraging variability\n7. Cognitive and technical burdens\n8. Variability's impact on quality\n9. Lifecycle integration\n10. Research agenda proposal\n11. Interdisciplinary dialogue\n12. Tools, techniques, theory advancements\n13. Roadmap for strategic directions\n14. Conference essence and goals

Now, please provide the skeleton for the following question.
{question}
"""

task2skeleton = {
    "ctx": skeleton_extract_template_ctx,
    "email": skeleton_extract_template_email,
    "paper": skeleton_extract_template_paper
}
task2data = {
    "ctx": "../../data/ablation/ctx_test.json",
    "paper": "../../data/ablation/paper_test.json",
}

def request_model(task, mode, split, tokenizer, llm, params, model_name):
    data = srsly.read_json(task2data[task])
    
    if "qwen" in model_name.lower() :
        template = QwenTemplate()
    elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
        template = MistralTemplate()
    elif "deepseek" in model_name.lower():
        template = DeepSeekTemplate()
    elif "stable" in model_name.lower():
        template = StableLMTemplate()
    elif "tinyllama" in model_name.lower():
        template = TinyLlamaTemplate()
    elif "h2o" in model_name.lower():
        template = H2OTemplate()
    elif "llama" in model_name.lower():
        template = LlamaTemplate()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    params.stop = template.get_stop_words()
    
    prompts = []
    for sample in tqdm(data, desc="Requesting, task: %s, mode: %s" % (task, mode), disable=True):
        skeleton_prompt = task2skeleton[task].format(question=sample["conversations"][0]["value"])
        prompts.append(skeleton_prompt)

    prompt_token_ids = tokenizer(prompts).input_ids
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=params)

    for output, sample in zip(outputs, data):
        sample["outline"] = output.outputs[0].text
        sample["conversations"][0]["value"] += "\nYou can refer to the following outlines: " + output.outputs[0].text

    print(Fore.GREEN + f"{task}_{split}_{mode}_outline_{model_name}.json" + Style.RESET_ALL)

    if os.path.exists("outputs") == False:
        os.mkdir("outputs")
    srsly.write_json(f"outputs/{task}_{split}_{mode}_outline_{model_name}.json", data)

def main(model_name_or_path, model_name, tp_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(model=model_name_or_path, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

    for task in ["ctx", "paper", "email"]:
        request_model(task, "without", "test", tokenizer, llm, sampling_params, model_name)
    
if __name__ == "__main__":
    Fire(main)