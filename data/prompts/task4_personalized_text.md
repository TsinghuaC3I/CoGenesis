# Task Overview
As an AI assistant, your task is to generate personalized text based on specified guidelines. Follow these instructions:

1. **Perspective**: Write from appropriate personal perspective, ensuring the text mirrors the user character's unique personality and private information.

2. **Realism and Personality**: The text should be authentic and coherent, vividly portraying the character's work and life.

3. **Writing Style**: Adapt the user's writing style in the text, focusing on creating a personalized and stylized content.

4. **Sequential Tasks**: Sequentially generate content for each task, ensuring that each piece is thematically connected or has related associations in content.

5. **Length**: The text should be a minimum of [[WORDS]] words.

6. **Output Format**: Provide the final text in JSON format, clearly labeled and structured.

# User Character Profile

[[PROFILE]]

# Task details for Generation:

[[TASK]]

# Expected Output Format (JSON)

```{
  "generated_text": [{
    "task": "The given task.",
    "details": "The details of given task."
    "task_id": "Task ID",
    "outline": ["Key point outline"],
    "style": ["Repeat writing style for personalized and stylized content generation"]
    "bullets": ["Privacy information and mobile phone activity logs included in the text, do not use the name as private information, include only the 3-5 most relevant bullets"],
    "length": "Text length, no less than [[WORDS]] words",
    "content": "Personalized text content"
  },
  // other tasks
  ]
}```
