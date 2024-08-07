#!/usr/bin/env python
from fastchat.conversation import Conversation, SeparatorStyle

class ConversationTemplate:
    def __init__(self):
        self.is_system_message = False
    
    def _get_prompt(self, prompt, system_prompt=None, role1=None, role2=None):
        conv = self.standard_conv.copy()
        if system_prompt is not None:
            conv.set_system_message(system_prompt)
        conv.append_message(role=role1, message=prompt)
        conv.append_message(role=role2, message=None)
        return conv.get_prompt()
    
    def copy(self):
        return self.standard_conv.copy()
    
    def __call__(self, prompt, system_prompt=None):
        pass
    
    def get_stop_words(self):
        pass

class MistralTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
    
    <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="mistral",
            system_template="[INST] {system_message}\n",
            roles=("[INST]", "[/INST]"),
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2="</s>",
        )
        
    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="[INST]", role2="[/INST]")
    
    def get_stop_words(self):
        return ["</s>", "[/INST]"]

class QwenTemplate(ConversationTemplate):
    """
    Refer to: https://github.com/QwenLM/Qwen
    https://github.com/QwenLM/Qwen1.5
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="qwen",
            system_template="<|im_start|>system\n{system_message}",
            system_message="You are a helpful assistant.",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
            stop_token_ids=[
                151643,
                151644,
                151645,
            ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
            stop_str="<|endoftext|>",
        )
        self.is_system_message = True
        
    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="<|im_start|>user", role2="<|im_start|>assistant")
    
    def get_stop_words(self):
        return ["<|endoftext|>", "<|im_end|>"]

    def get_stop_token_ids(self):
        return self.standard_conv.stop_token_ids

class DeepSeekTemplate(ConversationTemplate):
    """
    Refer to: https://github.com/deepseek-ai/DeepSeek-LLM
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="deepseek-chat",
            system_message="<｜begin▁of▁sentence｜>",  # must add a bos token before first message
            roles=("User", "Assistant"),
            sep_style=SeparatorStyle.DEEPSEEK_CHAT,
            sep="\n\n",
            sep2="<｜end▁of▁sentence｜>",
            stop_str="<｜end▁of▁sentence｜>",
        )
        
    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="User", role2="Assistant")
    
    def get_stop_words(self):
        return ["<｜end▁of▁sentence｜>"]

class StableLMTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/stabilityai/stablelm-zephyr-3b
    <|user|>
List 3 synonyms for the word "tiny"<|endoftext|>
<|assistant|>
1. Dwarf
2. Little
3. Petite<|endoftext|>
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="solar",
            system_message="",
            roles=("<|user|>", "<|assistant|>"),
            sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
            sep="<|endoftext|>\n",
            stop_str="<|endoftext|>",
        )

    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="<|user|>", role2="<|assistant|>")
    
    def get_stop_words(self):
        return ["<|endoftext|>"]

class OrcaPhiTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/stabilityai/stablelm-zephyr-3b
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="oo-phi-1_5",
            system_template="<|im_start|>system\n{system_message}",
            system_message="You are a helpful assistant.",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
            stop_token_ids=[
                50256,50296,50295
            ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
            stop_str="<|endoftext|>",
        )
        self.is_system_message = True

    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="<|im_start|>user", role2="<|im_start|>assistant")
    
    def get_stop_words(self):
        return ["<|endoftext|>"]

class TinyLlamaTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/stabilityai/stablelm-zephyr-3b
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="TinyLlama",
            system_template="<|system|>\n{system_message}",
            roles=("<|user|>", "<|assistant|>"),
            sep_style=SeparatorStyle.CHATML,
            sep="</s>",
            stop_token_ids=[2],
            stop_str="</s>",
        )
        self.is_system_message = True

    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="<|user|>", role2="<|assistant|>")
    
    def get_stop_words(self):
        return ["</s>"]


class H2OTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/h2oai/h2o-danube-1.8b-chat
    # <|system|>You are a friendly chatbot</s><|prompt|>Why is drinking water so healthy?</s><|answer|> Drinking water is healthy for several reasons: [...]
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="h2ogpt",
            system_template="<|system|>{system_message}</s>",
            roles=("<|prompt|>", "<|answer|>"),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="</s>",
        )
        self.is_system_message = True

    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="<|prompt|>", role2="<|answer|>")
    
    def get_stop_words(self):
        return ["</s>"]

class LlamaTemplate(ConversationTemplate):
    """
    Refer to: https://huggingface.co/h2oai/h2o-danube-1.8b-chat
    # <|system|>You are a friendly chatbot</s><|prompt|>Why is drinking water so healthy?</s><|answer|> Drinking water is healthy for several reasons: [...]
    """
    def __init__(self):
        super().__init__()
        self.standard_conv = Conversation(
            name="llama-2",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            roles=("[INST]", "[/INST]"),
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
        self.is_system_message = True

    def __call__(self, prompt, system_prompt=None):
        return self._get_prompt(prompt, system_prompt, role1="[INST]", role2="[/INST]")
    
    def get_stop_words(self):
        return ["</s><s>", "</s>"]

if __name__ == "__main__":
    template = MistralTemplate()
    prompt = template("Hello, world!")
    print("MistralTemplate: ")
    print(prompt)
    print("\n")
    
    template = QwenTemplate()
    prompt = template("Hello, world!", "Always say hello.")
    print("QwenTemplate: ")
    print(prompt)
    print("\n")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/pubmodels/transformers/Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)
    prompt = "Hello, world!"
    messages = [
        {"role": "system", "content": "Always say hello."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    
    template = DeepSeekTemplate()
    prompt = template("Hello, world!")
    print("DeepSeekTemplate: ")
    print(prompt)
    print("\n")
    
    tokenizer = AutoTokenizer.from_pretrained("/root/pubmodels/transformers/deepseek-ai/deepseek-llm-67b-chat", trust_remote_code=True)
    prompt = "Hello, world!"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    print(tokenizer.eos_token)
    print("\n")

    template = StableLMTemplate()
    prompt = template("Hello, world!")
    print("StableLMTemplate: ")
    print(prompt)
    print("\n")
    

    template = OrcaPhiTemplate()
    prompt = template("Hello, world!")
    print("OrcaPhiTemplate: ")
    print(prompt)
    print("\n")

    template = TinyLlamaTemplate()
    prompt = template("Hello, world!", "You are AI.")
    print("TinyLlamaTemplate: ")
    print(prompt)
    print("\n")
    
    template = H2OTemplate()
    prompt = template("Hello, world!", "You are AI.")
    print("H2OTemplate: ")
    print(prompt)
    print("\n")
    
    tokenizer = AutoTokenizer.from_pretrained("/root/pubmodels/transformers/h2oai/h2o-danube-1.8b-chat", trust_remote_code=True)
    prompt = "Hello, world!"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    print(tokenizer.eos_token)
    print("\n")

    template = LlamaTemplate()
    prompt = template("Hello, world!", "You are AI.")
    print("LlamaTemplate: ")
    print(prompt)
    print("\n")
    