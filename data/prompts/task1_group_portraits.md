# Task Overview
As an AI assistant, your task is to construct extremely diverse user group portraits in real-world. Follow these guidelines to ensure diversity and realism:

1. **Demographics**: Include a range of ages, professions, and hobbies in each portrait to represent a broad spectrum of users.

2. **Real-World Scenarios**: Make sure these portraits reflect a variety of real-world uses for an AI assistant. Focus particularly on scenarios involving personalized text writing, such as crafting emails, speeches, blogs, etc.

3. **Detailed Composition**: For each group portrait, include:
   - **Age Range**: Specify the age range for each user group. For example, "18-24 years", "30-40 years", etc.

   - **Professional Field**: Identify the professional field or occupation of the user group. Examples include "Software Developer", "High School Teacher", "Graduate Student", etc.

   - **Interests**: List relevant usual work, hobbies and interests for each group. Aim for a variety of interests that reflect the group's characteristics.

   - **AI Assistant Usage**: Outline specific scenarios where the group would use an AI assistant for personalized and creative text writing in daily work and life. These could include "drafting business emails", "writing creative blogs", "composing academic papers", "write a long tweet" etc.

4. **Quantity and Diversity**: Create a total of [[NUMBER]] unique and highly diversified group portraits.

5. **Output Format**: Structure your output as an array of JSON objects, each representing a different user group. Provide at least [[NUMBER]] varied and comprehensive group portraits.

Aim for detailed and imaginative portraits that could feasibly represent a wide range of potential AI assistant users.

# Expected Output Format (JSON)
```
[{
  "age_range": "Enter specific age range",
  "professional_field": "Enter professional field",
  "interests": ["List of diverse hobbies"],
  "works": ["List of diverse works"],
  "ai_assistant_usage": ["Diverse scenarios related to personalized content writing"]
},
{
  "age_range": "Enter specific age range",
  "professional_field": "Enter professional field",
  "interests": ["List of diverse hobbies"],
  "works": ["List of diverse works"],
  "ai_assistant_usage": ["Diverse scenarios related to personalized content writing"]
},
// Continue with additional user groups
]```