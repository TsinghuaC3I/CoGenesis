#!/usr/bin/env python
import os
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_API_BASE"]

class LLMs:
    def __init__(self, model="gpt-4-1106-preview", request_type="openai", parameters={"top_p": 0.7, "temperature": 0.9}):
        self.model = model
        self.request_type = request_type

        assert request_type == "openai"
        
        self.client = ChatOpenAI(model_name=model)
        self.client.model_kwargs = parameters

    def request(self, prompt):
        try:
            batch_messages = [[
                HumanMessage(content=prompt),
            ]]

            results = self.client.generate(batch_messages)
            model_output = results.generations[0][0].text
            return model_output
        except Exception as e:
            print(e)
            return None

if __name__ == "__main__":
    pass