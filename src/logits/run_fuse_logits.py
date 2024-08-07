#!/usr/bin/env python
from fire import Fire
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig

MIN_VALUE = 0
# MIN_VALUE = -1e9 will cause the model to output -inf, and the loss will be fixed

class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.length = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {key: tensor[idx] for key, tensor in self.data_dict.items()}
        return sample

class ProbFusionModel(nn.Module):
    def __init__(self, vocab_size):
        super(ProbFusionModel, self).__init__()
        self.fc1 = nn.Linear(vocab_size*2, 512, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 16, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1, bias=True)
        self.sigmoid =  nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'fc3' not in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'weight' in name and 'fc3' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, logits1, logits2):
        combined_logits = torch.cat([logits1, logits2], dim=-1)

        x = self.fc1(combined_logits)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        weights = self.sigmoid(x)
        fused_logits = logits1 * weights + logits2 * (1-weights)
        
        return fused_logits, weights

def load_logits_data(task, model, split, mode):
    data = torch.load(f"outputs/{task}_{split}_{mode}_{model}.pt")
    return data

def find_position(labels):
    for i, label in enumerate(labels):
        if label != -100:
            return i

def create_token_logits(token_logits, pad_value=0, pad_size=15):
    # empty = torch.zeros(vocab_size, dtype=torch.float)
    # keys = list(token_logits.keys())
    # empty[keys] = torch.tensor(list(token_logits.values()), dtype=torch.float)
    # return empty
    pad_size = pad_size - len(token_logits)
    keys = list(token_logits.keys()) + [pad_value] * pad_size
    values = list(token_logits.values()) + [MIN_VALUE] * pad_size
    
    # keys_tensor = torch.tensor(keys, dtype=torch.int)
    # values_tensor = torch.tensor(values, dtype=torch.float)
    return [keys, values]

def preprocess(data_small, data_large, vocab_size, max_length=1024):
    labels_small = data_small["labels"]
    labels_large = data_large["labels"]
    assert len(labels_small) == len(labels_large), "The labels are not consistent. {} != {}".format(len(labels_small), len(labels_large))
    logits_small = data_small["logits"]
    logits_large = data_large["logits"]
    assert len(logits_small) == len(logits_large), "The logits are not consistent. {} != {}".format(len(logits_small), len(logits_large))
    
    final_small = []
    final_large = []
    final_labels = []
    pad_value = vocab_size - 1
    pad_size = 16
    # empty_key = torch.zeros(pad_value, dtype=torch.int)
    # empty_value = torch.zeros(pad_value, dtype=torch.float)
    empty_key = [pad_value] * pad_size
    empty_value = [MIN_VALUE] * pad_size
    empty_arr = [empty_key, empty_value]
    for los, lol, las, lal in tqdm(zip(logits_small, logits_large, labels_small, labels_large), total=len(labels_small), desc="Preprocessing"):
        pos_s = find_position(las)
        pos_l = find_position(lal)
        assert len(las) - pos_s == len(lal) - pos_l, "The labels are not consistent. {} != {}".format(len(las) - pos_s, len(lal) - pos_l)
        # assert all(s == l for s, l in zip(las[pos_s:], lal[pos_l:])), "The labels are not consistent. {} != {}".format(las[pos_s:], lal[pos_l:])
        seq_logit_s = [create_token_logits(logit, pad_value=pad_value, pad_size=pad_size) for logit in los[pos_s:]]
        seq_logit_l = [create_token_logits(logit, pad_value=pad_value, pad_size=pad_size) for logit in lol[pos_l:]]
        las = las[pos_s:]
        padding = max_length - len(seq_logit_s)
        assert len(seq_logit_s) == len(seq_logit_l) == len(las), "The logits are not consistent. {} != {} != {}".format(len(seq_logit_s), len(seq_logit_l), len(las))
        if len(seq_logit_s) < max_length:
            seq_logit_s = seq_logit_s + [empty_arr for _ in range(padding)]
            seq_logit_l = seq_logit_l + [empty_arr for _ in range(padding)]
            final_labels.append(las + [-100]*padding)
        else:
            seq_logit_s = seq_logit_s[:max_length]
            seq_logit_l = seq_logit_l[:max_length]
            final_labels.append(las[:max_length])
        final_small.append(seq_logit_s)
        final_large.append(seq_logit_l)
    
    print("Preprocessing finished")
    final_small_logits = torch.tensor([[vv[1] for vv in v] for v in final_small], dtype=torch.float)
    final_small_indices = torch.tensor([[vv[0] for vv in v] for v in final_small], dtype=torch.long)
    final_large_logits = torch.tensor([[vv[1] for vv in v] for v in final_large], dtype=torch.float)
    final_large_indices = torch.tensor([[vv[0] for vv in v] for v in final_large], dtype=torch.long)
    final_labels = torch.tensor(final_labels)
    assert len(final_small_logits) == len(final_large_logits), "The logits are not consistent. {} != {}".format(len(final_small_logits), len(final_large_logits))
    assert len(final_small_logits) == len(final_labels), "The labels are not consistent. {} != {}".format(len(final_small_logits), len(final_labels))
    return {
        "logits_small": final_small_logits,
        "logits_large": final_large_logits,
        "labels": final_labels,
        "indices_small": final_small_indices,
        "indices_large": final_large_indices
    }

class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps))
        )

def main(task="ctx", small_model="Qwen-1_8B-Chat", large_model="Qwen-72B-Chat", small_config="Qwen/Qwen-1_8B-Chat"):
    print(f"""
          task = {task}
          small_model = {small_model}
          large_model = {large_model}
          small_config = {small_config}
          """)
    config_small = AutoConfig.from_pretrained(small_config, trust_remote_code=True)
    train_small = load_logits_data(task, small_model, "train", "with")
    train_large = load_logits_data(task, large_model, "train", "without")
    dev_small = load_logits_data(task, small_model, "dev", "with")
    dev_large = load_logits_data(task, large_model, "dev", "without")
    
    print("Preprocessing training data")
    train_inputs = preprocess(train_small, train_large, config_small.vocab_size)
    print("Preprocessing dev data")
    dev_inputs = preprocess(dev_small, dev_large, config_small.vocab_size)
    print("Save data")
    # torch.save(train_inputs, f"logits/{task}_{small_model}_{large_model}_train.pt")
    # torch.save(dev_inputs, f"logits/{task}_{small_model}_{large_model}_dev.pt")
    
    print("Prepare dataloader")
    train_dataset = DictDataset(train_inputs)
    dev_dataset = DictDataset(dev_inputs)
    
    epochs = 4
    batch_size = 2
    warmup_steps = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    print("Initialize model")
    model = ProbFusionModel(config_small.vocab_size)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    total_steps = len(train_dataloader) * epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    best_loss = np.inf
    for epoch in range(epochs):
        step = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            logits1 = batch["logits_small"].cuda()
            logits2 = batch["logits_large"].cuda()
            indices1 = batch["indices_small"].cuda()
            indices2 = batch["indices_large"].cuda()
            targets = batch["labels"].cuda()

            bsz, sql, _ = logits1.size()
            # logits1_full =  torch.zeros(bsz, sql, config_small.vocab_size, device=logits1.device)
            logits1_full = torch.full((bsz, sql, config_small.vocab_size), MIN_VALUE, device=logits1.device, dtype=torch.float)
            logits1_full.scatter_(2, indices1, logits1)
            # logits2_full =  torch.zeros(bsz, sql, config_small.vocab_size, device=logits2.device)
            logits2_full = torch.full((bsz, sql, config_small.vocab_size), MIN_VALUE, device=logits2.device, dtype=torch.float)
            logits2_full.scatter_(2, indices2, logits2)
            optimizer.zero_grad()
            outputs, _ = model(logits1_full, logits2_full)
            loss = criterion(outputs.view(-1, config_small.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                eval_loss = 0
                flag = True
                model.eval()
                for eval_batch in dev_dataloader:
                    logits1 = eval_batch["logits_small"].cuda()
                    logits2 = eval_batch["logits_large"].cuda()
                    indices1 = eval_batch["indices_small"].cuda()
                    indices2 = eval_batch["indices_large"].cuda()
                    targets = eval_batch["labels"].cuda()

                    bsz, sql, _ = logits1.size()
                    logits1_full = torch.full((bsz, sql, config_small.vocab_size), MIN_VALUE, device=logits1.device, dtype=torch.float)
                    logits1_full.scatter_(2, indices1, logits1)
                    # logits2_full =  torch.zeros(bsz, sql, config_small.vocab_size, device=logits2.device)
                    logits2_full = torch.full((bsz, sql, config_small.vocab_size), MIN_VALUE, device=logits2.device, dtype=torch.float)
                    logits2_full.scatter_(2, indices2, logits2)
                    outputs, weights = model(logits1_full, logits2_full)
                    loss = criterion(outputs.view(-1, config_small.vocab_size), targets.view(-1))
                    eval_loss += loss.item()
                    # if flag:
                    #     print(weights[:, :2, :])
                    # flag = False
                print("Eval loss: {}".format(eval_loss / len(dev_dataloader)))
                model.train()
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), f"outputs/{task}/{small_model}_{large_model}_best.pt")
            step += 1

        print("save model")
        torch.save(model.state_dict(), f"outputs/{task}/{small_model}_{large_model}_epoch{epoch}.pt")

if __name__ == '__main__':
    Fire(main)
