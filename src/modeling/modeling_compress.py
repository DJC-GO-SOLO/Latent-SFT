import logging
import json
from dataclasses import dataclass
from typing import Dict, Optional
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from torch.distributions import MultivariateNormal

import torch.distributed as dist

logger = logging.getLogger(__name__)

@dataclass
class CompressOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None

def softmax_over_embedding(
    x: torch.Tensor,                 # [seq, h]
    embedding: nn.Embedding,         # weight: [vocab, h]
    temperature: float = 1.0,
    use_cosine: bool = False,
    eps: float = 1e-12,
):
    W = embedding.weight.detach()                     # [vocab, h]
    x = x.to(dtype=W.dtype, device=W.device)

    if use_cosine:
        x_n = F.normalize(x, p=2, dim=-1, eps=eps)
        W_n = F.normalize(W, p=2, dim=-1, eps=eps)
        logits = F.linear(x_n, W_n)          # [seq, vocab]
    else:
        logits = F.linear(x, W)              # [seq, vocab]

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits.float(), dim=-1).to(W.dtype)  # [seq, vocab]
    new_x = probs @ W                                          # [seq, h]

    return new_x, probs

class LatentSFTStage1Encoder(nn.Module):
    def __init__(self,
        encoder_name_or_path: str = None,  
        decoder_name_or_path: str = None,
        bfloat16: bool = False, 
        use_flash_attention_2: bool = False, 
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else 'sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path)
        self.compress_token = '<|compress_token|>'  
        
        if training: 
            if self.tokenizer.pad_token is None:
                if 'deepseek' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token = self.tokenizer.eos_token  # <|endoftext|>
                elif 'llama' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token_id = 128001 # eos_token appears in LLaMA 3.2's command template, so a different token is used for padding
                else:
                    raise ValueError("Unsupported model type")
        
            special_tokens_dict = {'additional_special_tokens': [self.compress_token]} 
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
            if dist.get_rank() == 0 and save_path is not None:
                self.save(os.path.join(save_path, 'base_model'))
            
            
       
        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)
        
        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
        
        self.lora_tune = lora_tune
        self.save_path = save_path
        

        if lora_tune:
            if lora_path is not None:
                self.encoder = PeftModel.from_pretrained(
                    self.encoder, lora_path
            )
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                
                self.encoder = get_peft_model(self.encoder, self.config)

         
        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else 'sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )
        self.init_decoder()
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = self.encoder.config
        

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable(**kwargs)


    def _compress(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, 
    ):
        
        compress_outputs = self.encoder(input_ids, attention_mask = attention_mask)

        return compress_outputs.last_hidden_state


    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None, 
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape
        
        compress_mask = (cot_ids ==  self.compress_token_id)
        
        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor
        input_ids = input_ids.clone()                   
        input_ids[decoder_mask] = self.tokenizer.pad_token_id # Temporarily use pad_token_id as a placeholder
        base = self.decoder.model.embed_tokens(input_ids).detach()  # [bs, seq, h], requires_grad=False
        inputs_embeds = base.clone()  # Clone a shell

        
        bs, seq_len, h = base.shape

        # === Scatter new_b back to [seq, h], stacking sample-wise ===
        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]
            if compress_idx.numel() == 0:
                # No replacement needed; use the base directly
                emb_list.append(base[b])
                continue

            assert compress_idx.shape[0] == decoder_idx.shape[0], \
                f"Sample {b}: mismatch between number of compress_ids and -100 labels"

            # 1) Extract the segment to be mapped to the vocabulary (from encoder, requires gradient flow)
            x_b = compress_embedding[b, compress_idx]          # [m, h]

            # 2) Softmax over decoder's embedding.weight → expected embedding
            new_b, _ = softmax_over_embedding(
                x_b,
                self.decoder.model.embed_tokens,               
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h] requires gradient

            # 3) Use scatter with row indices to write new_b back into [seq, h]
            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = base[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, 
            attention_mask = attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = self.loss_fct(shift_logits, shift_labels)

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.encoder.save_pretrained(output_dir, state_dict=state_dict)




class LatentSFTStage1Decoder(nn.Module):
    def __init__(self,
        encoder_name_or_path: str = None,  
        decoder_name_or_path: str = None,
        bfloat16: bool = False, 
        use_flash_attention_2: bool = False, 
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='sdpa',#'flash_attention_2' if use_flash_attention_2 else 'sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        self.freeze_model(self.encoder)
        self.encoder.eval()
        
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path )
        self.compress_token = '<|compress_token|>'   
        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else 'sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )

        
        if training: 
            if self.tokenizer.pad_token is None:
                if 'deepseek' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token = self.tokenizer.eos_token  # <|endoftext|>
                elif 'llama' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token_id = 128001 # eos_token appears in LLaMA 3.2's command template, so a different token is used for padding
                else:
                    raise ValueError("Unsupported model type")
           
            if dist.get_rank() == 0 and save_path is not None:
                self.save(os.path.join(save_path, 'base_model'))
            
            
        
        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)
        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
    
        
        self.lora_tune = lora_tune
        self.save_path = save_path
        

        if lora_tune:
            if lora_path is not None:
                self.decoder = PeftModel.from_pretrained(
                    self.decoder, lora_path
            )
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                self.decoder = get_peft_model(self.decoder, self.config)

        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = self.encoder.config


    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.decoder.config.use_cache = False
        self.decoder.enable_input_require_grads()
        self.decoder.gradient_checkpointing_enable(**kwargs)

    def _compress(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, 
    ):
        
        compress_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        return compress_outputs.last_hidden_state


    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None, 
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape
        
        compress_mask = (cot_ids ==  self.compress_token_id) # bs:seq torch.BoolTensor
       
        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor
        
        input_ids[decoder_mask] = self.tokenizer.pad_token_id 
        inputs_embeds = self.decoder.model.model.embed_tokens(input_ids) 

        
        bs, seq_len, h = inputs_embeds.shape

        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]
            if compress_idx.numel() == 0:
                emb_list.append(inputs_embeds[b])
                continue

            assert compress_idx.shape[0] == decoder_idx.shape[0], \
                f"Sample {b}: mismatch between number of compress_ids and -100 labels"


            new_b, _ = softmax_over_embedding(
                compress_embedding[b, compress_idx]  ,
                self.decoder.model.model.embed_tokens,              
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h]

            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = inputs_embeds[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, 
            attention_mask = attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = self.loss_fct(shift_logits, shift_labels)

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.decoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.decoder.save_pretrained(output_dir, state_dict=state_dict)


class LatentSFTStage1Union(nn.Module):
    def __init__(self,
        encoder_name_or_path: str = None,  
        decoder_name_or_path: str = None,
        bfloat16: bool = False, 
        use_flash_attention_2: bool = False, 
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else 'sdpa',
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name_or_path )
        self.compress_token = '<|compress_token|>'   
        self.decoder = AutoModelForCausalLM.from_pretrained(  self.decoder_name_or_path, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else 'sdpa',
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )

        
        if training: 
            if self.tokenizer.pad_token is None:
                if 'deepseek' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token = self.tokenizer.eos_token  # <|endoftext|>
                elif 'llama' in decoder_name_or_path.lower():
                    self.tokenizer.pad_token_id = 128001 # eos_token appears in LLaMA 3.2's command template, so a different token is used for padding
                else:
                    raise ValueError("Unsupported model type")
            
        self.compress_token_id = self.tokenizer.convert_tokens_to_ids(self.compress_token)
        self.latent_token_ids = self.tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
        
        self.lora_tune = lora_tune
        self.save_path = save_path
        

        if lora_tune:
            if lora_path is not None:
                self.encoder = PeftModel.from_pretrained(
                    self.encoder, os.path.join(lora_path,'encoder_weight')
            )
                self.decoder = PeftModel.from_pretrained(
                    self.decoder, os.path.join(lora_path,'decoder_weight')
            )
                
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                
                self.encoder = get_peft_model(self.encoder, self.config)

                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                
                self.decoder = get_peft_model(self.decoder, self.config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable(**kwargs)

        self.decoder.config.use_cache = False
        self.decoder.enable_input_require_grads()
        self.decoder.gradient_checkpointing_enable(**kwargs)

        


    def _compress(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        
        compress_outputs = self.encoder(input_ids, attention_mask = attention_mask)

        return compress_outputs.last_hidden_state


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cot_ids: torch.LongTensor = None, 
        cot_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding = self._compress(cot_ids, cot_attention_mask)
        bs, _, _ = compress_embedding.shape
        
        compress_mask = (cot_ids ==  self.compress_token_id) # bs:seq torch.BoolTensor
        
        decoder_mask = (input_ids == -100)  #bs:seq torch.BoolTensor

        input_ids[decoder_mask] = self.tokenizer.pad_token_id 
        base = self.decoder.model.model.embed_tokens(input_ids) # [bs, seq, h], requires_grad=False
        inputs_embeds = base.clone()  

        
        bs, seq_len, h = inputs_embeds.shape

        emb_list = []
        for b in range(bs):
            compress_idx = compress_mask[b].nonzero(as_tuple=False).squeeze(-1)   # [m]
            decoder_idx  = decoder_mask[b].nonzero(as_tuple=False).squeeze(-1)    # [m]
            
            assert compress_idx.shape[0] == decoder_idx.shape[0], \
                f"Sample {b}: mismatch between number of compress_ids and -100 labels"

            new_b, _ = softmax_over_embedding(
                compress_embedding[b, compress_idx]  ,
                self.decoder.model.model.embed_tokens,              
                temperature=1.0,
                use_cosine=False
            )                                                  # [m, h]

            idx_exp = decoder_idx.unsqueeze(-1).expand(-1, h)  # [m, h]
            replaced = base[b].scatter(0, idx_exp, new_b)      # [seq, h]

            emb_list.append(replaced)

        inputs_embeds = torch.stack(emb_list, dim=0)           # [bs, seq, h]

        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, 
            attention_mask = attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = self.loss_fct(shift_logits, shift_labels)

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return CompressOutput(
            loss=loss,
            logits=logits,
        )
    def save(self, output_dir: str):
        state_dict = self.encoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.encoder.save_pretrained(output_dir+'/encoder_weight', state_dict=state_dict)

        state_dict = self.decoder.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.decoder.save_pretrained(output_dir+'/decoder_weight', state_dict=state_dict)
