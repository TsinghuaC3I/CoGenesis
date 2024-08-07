#!/usr/bin/env python

import os
import srsly
from datetime import datetime
from fire import Fire
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from ..llms_oai import LLMs

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
    "paper": skeleton_extract_template_paper
}
task2data = {
    "ctx": "../../data/ablation/ctx_test.json",
    "paper": "../../data/ablation/paper_test.json",
}

def threaded_request(llms_instance, rdata, task):
    skeleton_extract_template = task2skeleton[task]
    prompt = skeleton_extract_template.format(question=rdata["content"])
    rdata["result"] = llms_instance.request(prompt)
    return rdata

def request_model(task, mode="without", split="train", model_name="gpt-3.5-turbo-1106"):
    data = srsly.read_json(task2data[task])
    
    # llm = LLMs(model="gpt-4-1106-preview", request_type="openai")
    llm = LLMs(model=model_name, request_type="openai")

    num_threads = 200  # number of concurrent threads
    contents = []
    for sample in data:
        content = sample["conversations"][0]["value"]
        contents.append({"content": content, "tid": sample["tid"]})

    print(f"Requesting {len(contents)} contents with {num_threads} threads")
    start_time = datetime.now()
    # Using ThreadPoolExecutor to manage a pool of threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
       # Submit tasks to the executor and store future objects
        futures = [executor.submit(threaded_request, llm, content, task) for content in contents]

        # Collect results as they become available
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'An exception occurred: {exc}')
                # results.append(None)

    print("Requesting finished")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time}")

    None_ids = []
    tid2result = {result["tid"]:result for result in results}
    for sample in data:
        if sample["tid"] in tid2result and tid2result[sample["tid"]]["result"] is not None:
            sample["outline"] = tid2result[sample["tid"]]["result"]
            sample["conversations"][0]["value"] += "\nYou can refer to the following outlines: " + tid2result[sample["tid"]]["result"]
        else:
            None_ids.append(sample["tid"])
    print(task, mode, split, "None_ids:", len(None_ids), None_ids)

    if os.path.exists("outputs") == False:
        os.makedirs("outputs")
    srsly.write_json(f"outputs/{task}_{split}_{mode}_outline_{model_name}.json", data)

if __name__ == "__main__":
    Fire(request_model)