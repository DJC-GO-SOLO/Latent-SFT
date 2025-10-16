import os
import json
import sglang as sgl
import torch.multiprocessing as mp
from transformers import AutoTokenizer


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def get_last_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(
        text,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]
    if not ids:
        raise ValueError(f"Tokenization produced empty ids for marker: {text!r}")
    return ids[0]  


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  

    # ======== Config ========
    data_path = "../data/GSM8k-Aug-test.jsonl"
    model_path = "../output/stage2results/llama3.2-1b-stage2/best_hf"
    out_path = "../output/stage2results/llama3.2-1b-stage2/best_hf/results.jsonl"

    data = read_jsonl(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    latent_end_token_id = get_last_token_id(tokenizer, "</think>")

    prompts = []
    for item in data:
        if 'llama' in model_path.lower():
            input_text = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}.\n"
                f"{item['problem']}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "<think>"
            )
        elif 'deepseek' in model_path.lower():
            messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]
            input_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )

            input_text = input_text + "<｜Assistant｜>"
        else:
            raise ValueError("Model type not supported. Only 'llama' and 'deepseek' are supported.")

        input_ids = tokenizer(input_text,  truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False)["input_ids"]
        prompts.append(input_ids)

    
    llm = sgl.Engine(
        model_path=model_path,
        trust_remote_code=True,
        dtype="bfloat16",          
        kv_cache_dtype="auto",
        enable_latent=True,
        latent_end_token_id=latent_end_token_id,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        mem_fraction_static=0.8,
        tp_size=4,
        sampling_backend='flashinfer',
        max_running_requests=64,
        log_level="info",
        skip_tokenizer_init=True
    )

   
    outputs = llm.generate(
        input_ids = prompts,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 1024,   
        },
    )

    for idx, o in enumerate(outputs):
        data[idx]["prediction"] = tokenizer.decode(o['output_ids'], skip_special_tokens=False)
    write_jsonl(data, out_path)
    print(f"Done. Wrote {len(data)} lines to: {out_path}")


if __name__ == "__main__":
    main()
