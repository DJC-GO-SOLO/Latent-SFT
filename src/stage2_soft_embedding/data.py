import json
import torch
from torch.utils.data import Dataset
import re

def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data

class Stage2Dataset(Dataset):
    def __init__(
        self, 
        path,
        train_latent_soft_label_path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)
        
        self.train_latent_soft_label_path = train_latent_soft_label_path

        self.args = args
        self.model = model
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return pretrain_tokenize_function(
            examples = self.data[idx],
            latent_state = torch.load(self.train_latent_soft_label_path+f'/{idx}.pt'),
            model = self.model,
            idx=idx
        )


def pretrain_tokenize_function(examples, 
        model, 
        latent_state,
        idx
    ):
     # Caution: Since each model uses a different instruction template, we apply custom formatting 
    # instead of using `apply_chat_template` directly.
    if 'deepseek' in model.decoder_name_or_path.lower():
        messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]

        if '</think>' in examples["cot_answer"] or '</think>' in examples["problem"]:
            raise ValueError("</think> triggers template logic — needs revision")
        input_text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
        input_prefix = input_text + "<｜Assistant｜>"
        input_suffix = examples["cot_answer"] + "<｜end▁of▁sentence｜>"
    elif 'llama' in model.decoder_name_or_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"
        
        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_suffix = examples["cot_answer"] + "<|eot_id|>"
    else:
        raise ValueError("Unsupported model type")

    input_prefix_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']
    input_suffix_ids = model.tokenizer(input_suffix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']

    text_output = dict()

    text_output['input_ids'] =input_prefix_ids + model.latent_token_ids[0] + [-100] * len(latent_state)+ model.latent_token_ids[1] + input_suffix_ids
    text_output['labels'] = [-100] * len(input_prefix_ids)+ [-100] * len(model.latent_token_ids[0]) + [-100] * len(latent_state)+ model.latent_token_ids[1] +input_suffix_ids
    latent_start_index = len(input_prefix_ids  + model.latent_token_ids[0])
    latent_end_index = latent_start_index + len(latent_state) 

    text_output['latent_index'] = [latent_start_index, latent_end_index]
    text_output['latent_state']= latent_state
        
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        latent_index = [example["latent_index"] for example in examples]
        latent_state = [example["latent_state"] for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        
        labels = self.dynamic_padding(labels)

        batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "latent_state": latent_state,
            "latent_index": latent_index,
            "labels": labels}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences