
system_prompt_with_context_ctx = "You are now a helpful personal AI assistant. You should emulate the author's style and tone based on provided history content. Your responses should be detailed and informative, using the personal information reasonably in the user's profile. Aim for insightful and high-quality solutions that make users satisfied."

system_prompt_wo_context = "You are now a helpful personal AI assistant. Aim for insightful and high-quality solutions that make users satisfied."

########### prompts for context-aware instructions ###########
# few_shot prompts for request LLMs
ctx_few_shot = {
    "w_context": """## User Profile
{profile}

## User Writing History
{history}

## Task
{task}
""",
    "wo_context": "{task}"
}

########### prompts for personalized emails ###########
system_prompt_with_context_per = "You are now a helpful personal AI assistant. You should emulate the author's style and tone based on provided history content. Your responses should be detailed and informative, matching the author's unique writing approach. Aim for insightful and high-quality solutions that make users satisfied."

# few_shot prompts for request LLMs
email_few_shot = {
    "w_context": """## History Emails
{examples}

## Task
Compose an email for the subject '{task}' that matches the author's unique style and tone.
""",
    "wo_context": "Compose an email for the subject '{task}'"
}

########### prompts for personalized emails ###########
# few_shot prompts for request LLMs
paper_few_shot = {
    "w_context": """## History Paper Abstracts
{examples}

## Task
Compose an abstract for the title '{task}' that matches the author's unique content, style and tone.
""",
    "wo_context": "Compose an abstract for the title '{task}'"
}
