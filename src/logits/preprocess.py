#!/usr/bin/env python
IGNORE_TOKEN_ID = -100

def preprocess_qwen(
    user_prompts,
    assistant_prompts,
    system_prompts,
    tokenizer,
    max_len: int = 4096,
    is_padding=True):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    im_start = tokenizer("<|im_start|>").input_ids[0] #tokenizer.im_start_id
    im_end = tokenizer("<|im_end|>").input_ids[0] #tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    for user_prompt, assistant_prompt, system_prompt in zip(user_prompts, assistant_prompts, system_prompts):
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_prompt).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(system)
        assert len(input_id) == len(target)
        
        role = roles["user"]
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(user_prompt).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
        target += _target
        
        role = roles["assistant"]
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(assistant_prompt).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        _target = [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1) + _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
        target += _target

        assert len(input_id) == len(target)
        if is_padding:
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        else:
            input_ids.append(input_id)
            targets.append(target)

    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    # targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def preprocess_tinyllama(
    user_prompts,
    assistant_prompts,
    system_prompts,
    tokenizer,
    max_len: int = 4096,
    is_padding=True):
    system_tokens = tokenizer("<|system|>").input_ids
    roles = {"user": "<|user|>", "assistant": "<|assistant|>"}
    nl_tokens = tokenizer('\n').input_ids
    end_tokens = tokenizer("</s>").input_ids

    # Apply prompt templates
    input_ids, targets = [], []

    for user_prompt, assistant_prompt, system_prompt in zip(user_prompts, assistant_prompts, system_prompts):
        input_id, target = [], []
        system = system_tokens + nl_tokens + tokenizer(system_prompt).input_ids + end_tokens + nl_tokens
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(system)
        assert len(input_id) == len(target)
        
        role = roles["user"]
        role_tokens = tokenizer(role).input_ids
        _input_id = role_tokens + nl_tokens + \
                tokenizer(user_prompt).input_ids + end_tokens + nl_tokens
        input_id += _input_id
        _target = [IGNORE_TOKEN_ID] * len(_input_id)
        target += _target
        
        role = roles["assistant"]
        _input_id = role_tokens + nl_tokens + \
                tokenizer(assistant_prompt).input_ids + end_tokens
        input_id += _input_id
        _target = [IGNORE_TOKEN_ID] * len(role_tokens) + \
                    _input_id[len(role_tokens):]
        target += _target

        assert len(input_id) == len(target)
        if is_padding:
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        else:
            input_ids.append(input_id)
            targets.append(target)

    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    # targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
