import logging
import json
from dataclasses import dataclass
from typing import Dict, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training,TrainableTokensConfig
from transformers.file_utils import ModelOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from torch.distributions import Categorical
logger = logging.getLogger(__name__)

@dataclass
class LatentOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None

def kd_kl_loss(               # "True KL" × T², averaged over valid tokens
    logits: torch.Tensor,     # [B, S, V] student logits (pre-softmax)
    target_probs: torch.Tensor,  # [B, S, V] soft targets (teacher probabilities, possibly unnormalized)
    mask: torch.Tensor | None = None,  # [B, S] — 1 indicates valid tokens
    temperature: float = 1.0,
    eps: float = 1e-12,
):
    z = logits.float()
    if temperature != 1.0:
        z = z / temperature

    logp = F.log_softmax(z, dim=-1)                       # [B,S,V], float32

    q = target_probs.float()
    q = q / (q.sum(dim=-1, keepdim=True) + eps)           # Normalized teacher probabilities
    # KL(q||p) = sum q * (log q - log p)
    kl_per_tok = (q * (q.clamp_min(eps).log() - logp)).sum(dim=-1)  # [B,S]

    if mask is not None:
        m = mask.to(kl_per_tok.dtype)
        loss = (kl_per_tok * m).sum() / m.sum().clamp_min(1)
    else:
        loss = kl_per_tok.mean()

    return loss * (temperature ** 2)   # Classic KD: scale by T² to match gradient magnitude


class LatentSFTStage2SoftEmbedding(nn.Module):
    def __init__(self,
        latent_model_path: str = None,
        ce_w: float = 1.,
        kl_w: float = 1.,
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
        self.latent_model_path = latent_model_path
        
        self.latent_model = AutoModelForCausalLM.from_pretrained(  self.latent_model_path, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.latent_model_path)
        self.latent_tokens =  ['<think>','</think>']
        
        if training: 
            if self.tokenizer.pad_token is None:
                if 'deepseek' in latent_model_path.lower():
                    self.tokenizer.pad_token = self.tokenizer.eos_token  # <|endoftext|>
                elif 'llama' in latent_model_path.lower():
                    self.tokenizer.pad_token_id = 128001 # eos_token appears in LLaMA 3.2's command template, so a different token is used for padding
                else:
                    raise ValueError("Unsupported model type")
            
            if save_path is not None and dist.get_rank() == 0 :
                self.save(os.path.join(save_path, 'base_model'))
        

        self.latent_token_ids = self.tokenizer(self.latent_tokens, add_special_tokens=False)['input_ids']
        

        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.latent_model = PeftModel.from_pretrained(
                    self.latent_model, lora_path
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
                
                self.latent_model = get_peft_model(self.latent_model, self.config)

        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_w = ce_w
        self.kl_w = kl_w

    def gradient_checkpointing_enable(self, **kwargs):
        self.latent_model.config.use_cache = False
        self.latent_model.enable_input_require_grads()
        self.latent_model.gradient_checkpointing_enable(**kwargs)

    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None,
        latent_state: list = None, 
        latent_index: list = None, #B,2
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):  
        bs =  input_ids.shape[0]

        decoder_mask = (input_ids == -100)  
     
        ids_for_embed = input_ids.masked_fill(decoder_mask, self.tokenizer.pad_token_id)
       
        input_embeddings = self.latent_model.model.model.embed_tokens(ids_for_embed).detach() #bs:seq
        W = self.latent_model.model.model.embed_tokens.weight.detach()

        rows = []
        for b in range(bs):
            s, e = int(latent_index[b][0]), int(latent_index[b][1])
            x = latent_state[b]
            x = x.to(device=input_embeddings.device, dtype=input_embeddings.dtype)

            assert (e - s) == x.size(0), f"latent length mismatch: {(s, e)} vs {x.shape}"
            x = x @ W
            row = torch.cat([input_embeddings[b, :s], x, input_embeddings[b, e:]], dim=0)
            rows.append(row)
        input_embeddings = torch.stack(rows, dim=0)

        decoder_outputs = self.latent_model(inputs_embeds=input_embeddings, 
            attention_mask = attention_mask
        )

        logits = decoder_outputs.logits
        
        loss_ce = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.latent_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss_ce = self.loss_ce(shift_logits, shift_labels)

        loss_kl = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        for b in range(bs):
            s, e = int(latent_index[b][0]), int(latent_index[b][1])
            assert s >= 1 and e > s, f"bad latent_index: {(s, e)}"
            q = latent_state[b].to(device=logits.device, dtype=logits.dtype)  # [m,V]
            p_logits = logits[b, s-1:e-1]                                     # [m,V]
            loss_kl = loss_kl + kd_kl_loss(p_logits, q)                     # Standard soft cross-entropy (equivalent to KL + constant)

        loss_kl  = loss_kl / bs
        loss = self.ce_w * loss_ce + self.kl_w * loss_kl

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item(),
                        'loss_ce': loss_ce.item(),
                        'loss_kl': loss_kl.item(),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')

        return LatentOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.latent_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.latent_model.save_pretrained(output_dir, state_dict=state_dict)

    def one_example_generate_lora(
        self,
        inputs: dict,
        max_new_tokens=50,
        temperature=1.0, 
        top_p=0.9,
        do_sample=True,
    ):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        generated_ids = []
        past_key_values = None
        stop_triggered = False
        latent_states = []

        W = self.latent_model.model.model.embed_tokens.weight.detach()

        with torch.no_grad():
            #latent reasoning 
            for latent_step in range(max_new_tokens):
                if input_ids is not None:
                    input_embeddings = self.latent_model.model.model.embed_tokens(input_ids)

                outputs = self.latent_model(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                next_token_logits = outputs.logits[:, -1, :] #1:vab_size

                past_key_values = outputs.past_key_values
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                

                probs = torch.softmax(next_token_logits.float(), dim=-1).to(W.dtype)
                input_embeddings = probs @ W  
                input_embeddings = input_embeddings.unsqueeze(1)

                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.size(0), 1), 
                    device=attention_mask.device)
                ], dim=1)

                if next_token[0,0].item() == self.latent_token_ids[1][0]:
                    input_ids = next_token
                    break
                else:
                    input_ids = None
                    generated_ids.append(int(next_token.item()))
                    latent_states.append(input_embeddings)

            latent_state = torch.cat(latent_states, dim=1)


            think_ids = torch.LongTensor(self.latent_token_ids[1]).unsqueeze(0).to(self.latent_model.device)
            think_end_embeddings = self.latent_model.model.model.embed_tokens(think_ids)
            input_ids = inputs['input_ids'] 
            attention_mask = [1] * (input_ids.size(1)+latent_state.size(1)+len(self.latent_token_ids[1]))
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.latent_model.device).unsqueeze(0)

            input_embeddings = self.latent_model.model.model.embed_tokens(input_ids)
            input_embeddings = torch.cat([input_embeddings, latent_state, think_end_embeddings], dim=1)#
            generated_output = self.latent_model.model.generate(
                    inputs_embeds=input_embeddings,  
                    attention_mask=attention_mask,  
                    max_new_tokens=max_new_tokens,          
                    do_sample=do_sample,               
                    temperature=temperature,           
                    top_p=top_p,
                )
            if len(generated_ids) != 0:
                decoded_text_latent = model.tokenizer.decode(generated_ids, skip_special_tokens=False)
            else:
                decoded_text_latent = ""
            decoded_text = self.tokenizer.decode(generated_output[0], skip_special_tokens=False)
        
        generate_token_num = len(generated_ids)
        return {
            "text": decoded_text_latent + decoded_text,
            "stopped_early": stop_triggered,
            "generate_token_num": generate_token_num
        }
    
    @staticmethod
    def one_example_generate_hf(
        model,
        inputs: dict,
        max_new_tokens=50,
        temperature=1.0, 
        top_p=0.9,
        do_sample=True,
    ):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        generated_ids = []
        past_key_values = None
        stop_triggered = False

        latent_states = []

        W = model.model.embed_tokens.weight.detach()

        with torch.no_grad():
            #latent reasoning
            for latent_step in range(max_new_tokens):
                if input_ids is not None:
                    input_embeddings = model.model.embed_tokens(input_ids)

                outputs = model(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states = True
                )

                last_hidden_states = [i[:, -1, :] for i in outputs.hidden_states]

                next_token_logits = outputs.logits[:, -1, :] #1:vab_size

                past_key_values = outputs.past_key_values
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                

                probs = torch.softmax(next_token_logits, dim=-1) 
                input_embeddings = probs @ W  
 
                input_embeddings = input_embeddings.unsqueeze(1)

                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.size(0), 1), 
                    device=attention_mask.device)
                ], dim=1)

                if next_token[0,0].item() == model.latent_token_ids[1][0]:
                    input_ids = next_token
                    break
                else:
                    input_ids = None
                    generated_ids.append(int(next_token.item()))
                    latent_states.append(input_embeddings)
            
            latent_state = torch.cat(latent_states, dim=1)

            think_ids = torch.LongTensor(model.latent_token_ids[1]).unsqueeze(0).to(model.device)
            think_end_embeddings = model.model.embed_tokens(think_ids)
            input_ids = inputs['input_ids'] 
            attention_mask = [1] * (input_ids.size(1)+latent_state.size(1)+len(model.latent_token_ids[1]))
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(model.device).unsqueeze(0)

            input_embeddings = model.model.embed_tokens(input_ids)
            input_embeddings = torch.cat([input_embeddings, latent_state, think_end_embeddings], dim=1)
            generated_output = model.generate(
                    inputs_embeds=input_embeddings,  
                    attention_mask=attention_mask,  
                    max_new_tokens=max_new_tokens,           
                    do_sample=do_sample,              
                    temperature=temperature,           
                    top_p=top_p,
                )
          
            if len(generated_ids) != 0:
                decoded_text_latent = model.tokenizer.decode(generated_ids, skip_special_tokens=False)
            else:
                decoded_text_latent = ""
            decoded_text = model.tokenizer.decode(generated_output[0], skip_special_tokens=False)
        
        generate_token_num = len(generated_ids)
        return {
            "text": decoded_text_latent + decoded_text,
            "stopped_early": stop_triggered,
            "generate_token_num": generate_token_num
        }
