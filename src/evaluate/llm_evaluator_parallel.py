#!/usr/bin/env python
import os
import srsly
from tqdm import tqdm
from colorama import Fore, Style
from fire import Fire
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from ..llms_oai import LLMs
from ..prompts import ctx_few_shot, paper_few_shot, email_few_shot
task2few_shot = {
    "ctx": ctx_few_shot,
    "email": email_few_shot,
    "paper": paper_few_shot
}

prompt_overall_with_profile = """[Instruction]
Please act as an impartial evaluator and assess the quality of the AI assistant's response to the user question shown below. Your assessment should focus on how well the response aligns with the user's personalized profile and writing history. Evaluate factors such as the response's adherence to the user's personal style, consistency with their profile, helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[User Profile and Writing History]
{profile_info}
{writing_history}

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

prompt_personalize = """[Instruction]
Please act as an impartial judge an evaluate the AI assistant's response based on its alignment with the user's personal profile and writing history. Focus your assessment on the personalization aspects of the response, including its adherence to the user's unique style, preferences, and consistency with their profile. Consider how well the response addresses the user's individual needs and interests. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[User Profile and Writing History]
{profile_info}
{writing_history}

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

prompt_overall_without_profile = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

task2data = {
    "ctx": "/root/kyzhang/llms/CoGen/data_ctx/final_data/test.json",
    "email": "/root/kyzhang/llms/CoGen/data_per/final_data/avocado_test.json",
    "paper": "/root/kyzhang/llms/CoGen/data_per/final_data/paper_test.json"
}


model = {"model": "gpt-4-1106-preview", "request_type": "openai"}
llm = LLMs(**model)

def create_inputs(task, sample):
    profile_info = sample["additional_profile"] + "\n" if task == "ctx" else ""

    # task
    few_shot = task2few_shot[task]
    value = sample["conversations"][0]["value"]
    question = value if task == "ctx" else few_shot["wo_context"].format(task=value)
    
    # writing
    if task == "email":
        sample['profile'] = sample['profile'][:3]
    writing_history = "\n".join([f"Task: {few_shot['wo_context'].format(task=p['title'])}\nContent: {str(p['text'])}" for p in sample['profile']])
    
    return profile_info, writing_history, question

def preprocess_value(sample, task, model_name):
    value = sample["conversations"][0]["value"]
    if task == "email":
        ## Task\nCompose an email for the subject 'AvocadoIT Always on data sheet' that matches the author's unique style and tone.\n
        split_str = "Compose an email for the subject '"
        replace_str=  "' that matches the author's unique style and tone."
    elif task == "paper":
        ## Task\nCompose an abstract for the title 'Routing through networks with hierarchical topology aggregation' that matches the author's unique content, style and tone.\n
        split_str = "Compose an abstract for the title '"
        replace_str = "' that matches the author's unique content, style and tone."
    else:
        return sample["tid"]
    
    value = value.split(split_str)
    value = value[1].replace(replace_str, "").strip() if len(value) > 1 else value[0].replace("Title: ", "").strip()
    if "outline" in model_name:
        value = value.split("\n")[0].strip()

    return value

def evaluate_task(task, filename, model_name, save_folder="evaluation_golden", max_count=50):
    print(Fore.GREEN + f"Task: {task}, filename: {filename}, model_name: {model_name}, save_folder: {save_folder}, max_count: {max_count}" + Style.RESET_ALL)
    data = srsly.read_json(filename)[:max_count]
    golden_data = srsly.read_json(task2data[task])
    tid2sample = {sample["tid"]: sample for sample in golden_data}
    # title2idx = {preprocess_value(sample, task, model_name): sample["tid"] for sample in golden_data}
    for sample in data:
        sample["tid"] = str(sample["tid"])
    
    prompts = []
    for sample in data:
        if "model_output" not in sample or sample["model_output"] is None:
            continue
        
        output = sample["model_output"]
        if isinstance(output, dict):
            output = output[model_name]
        output = output.strip()
        
        # obtain profile, history, question from the golden_data
        profile_info, writing_history, question = create_inputs(task, tid2sample[sample["tid"]])
        
        # if mode == "overall":
        overall_prompt = prompt_overall_with_profile.format(
            profile_info=profile_info,
            writing_history=writing_history,
            question=question,
            answer=output
        )
        prompts.append({"prompt": overall_prompt, "model": model_name, "metric": "overall", "tid": sample["tid"]})

        # elif mode == "personalize":
        persona_prompt = prompt_personalize.format(
            profile_info=profile_info,
            writing_history=writing_history,
            question=question,
            answer=output
        )
        prompts.append({"prompt": persona_prompt, "model": model_name, "metric": "personalize", "tid": sample["tid"]})

        # elif mode == "overall_without_profile":
        overall_wo_prompt = prompt_overall_without_profile.format(
            question=question,
            answer=output
        )
        prompts.append({"prompt": overall_wo_prompt, "model": model_name, "metric": "overall_without_profile", "tid": sample["tid"]})

 
    num_threads = 200  # number of concurrent threads

    def threaded_request(llms_instance, rdata):
        rdata["result"] = llms_instance.request(rdata["prompt"])
        return rdata
    
    print(f"Requesting {len(prompts)} prompts with {num_threads} threads")
    start_time = datetime.now()
    # Using ThreadPoolExecutor to manage a pool of threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
       # Submit tasks to the executor and store future objects
        futures = [executor.submit(threaded_request, llm, prompt) for prompt in prompts]

        # Collect results as they become available
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'An exception occurred: {exc}')
                results.append(None)
    print("Requesting finished")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time}")

    tid2idx = {sample["tid"]: idx for idx, sample in enumerate(data)}
    for sample in data:
        sample["evaluation"] = {
            model_name: {
            "overall": None,
            "personalize": None,
            "overall_without_profile": None
        }}
        if isinstance(sample["model_output"], str):
            sample["model_output"] = {
                model_name: sample["model_output"]
            }

    num_not_none = 0
    for result in results:
        if result is not None:
            idx = tid2idx[result["tid"]]
            data[idx]["evaluation"][result["model"]][result["metric"]] = result["result"]
            num_not_none += 1
            
    print(f"Number of not None results: {num_not_none}/{len(results)}")

    now = datetime.now()
    datenote = now.strftime("%Y%m%d%H%M")
    suffix = filename.split("/")[-1]
    output_name = f"{task}_{datenote}_{suffix}"
    if os.path.exists(save_folder) == False:
        os.mkdir(save_folder)
    srsly.write_json(f"./{save_folder}/{output_name}", data)
    print(Fore.GREEN + f"Writing to ./{save_folder}/{output_name}" + Style.RESET_ALL)


if __name__ == "__main__":
    Fire(evaluate_task)
