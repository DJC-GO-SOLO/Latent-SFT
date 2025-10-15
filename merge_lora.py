
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path="../output/stage1decoderresults/llama3.2-1b-stage1-decoder/best_hf"
tokenizer_path="../output/stage1decoderresults/llama3.2-1b-stage1-decoder/best_hf"

lora_path="../output/stage1unionresults/llama3.2-1b-stage1-union/checkpoint-best/lora_adapter"
output_path='../output/stage1unionresults/llama3.2-1b-stage1-union/best_hf'


base_model = AutoModelForCausalLM.from_pretrained(base_model_path, 
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16 ,
            use_cache=False,
            trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = PeftModel.from_pretrained(
            base_model, lora_path
        )
model= model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)