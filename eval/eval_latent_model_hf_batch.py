import os
import random
import numpy as np
import torch
import fcntl


def seed_everything(seed: int = 777):
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


seed_everything(777)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.modeling.modeling_stage2 import LatentSFTStage2SoftEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import logging
import torch
import torch.multiprocessing as mp
import time
import argparse
from eval_utils.grader import check_is_correct
from eval_utils.parser import extract_answer


def prepare_single_example(examples, model, latent_model_path):
    """Tokenize and prepare a single example, returns dict with input_ids and attention_mask."""
    if 'deepseek' in latent_model_path.lower():
        messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]

        input_text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )

        input_prefix = input_text + "<｜Assistant｜>"
    elif 'llama' in latent_model_path.lower():
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{examples['problem']}<|eot_id|>"

        input_prefix = input_text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

    else:
        # raise ValueError("Unsupported model type")
        messages = [
                    {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + examples["problem"]},
                ]

        input_prefix = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
    

    input_ids = model.tokenizer(input_prefix, truncation=False, padding=False, add_special_tokens = False, return_attention_mask=False)['input_ids']
   
    text_input = dict()
    text_input['input_ids'] = input_ids+model.latent_token_ids[0]
    text_input['attention_mask'] = [1] * len(text_input['input_ids'])

    text_input['input_ids'] = torch.tensor(text_input['input_ids'], dtype=torch.long).to(model.device).unsqueeze(0)
    text_input['attention_mask'] = torch.tensor(text_input['attention_mask'], dtype=torch.long).to(model.device).unsqueeze(0)

    return text_input


class MultiprocessTransformerWrapper:
    def __init__(
        self,
        latent_model_path,
        mp_size=1,
        dtype='float16',
        max_length=4096,
        max_new_tokens=128,
        topk_interpolation=10,
        # Gumbel noise parameters
        add_gumbel_noise=False,
        gumbel_temperature=1.0,
        noise_scale=1.0,
        batch_size=1,
    ):
        self.latent_model_path = latent_model_path
        self.mp_size = mp_size
        self.dtype = dtype
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.topk_interpolation = topk_interpolation

        self.add_gumbel_noise = add_gumbel_noise
        self.gumbel_temperature = gumbel_temperature
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=MultiprocessTransformerWrapper._gen_per_process, 
                args=(
                    self.latent_model_path,
                    self.dtype, 
                    rank, 
                    self.input_queue, 
                    self.output_queue,
                    self.max_length,
                    self.max_new_tokens,
                    self.topk_interpolation,
                    self.add_gumbel_noise,
                    self.gumbel_temperature,
                    self.noise_scale,
                )
            )
            p.start()
            self.processes.append(p)
        
        self.init_timer()
            

    def close(self):
        for p in self.processes:
            p.terminate()
        
        for p in self.processes:
            p.join()
            p.close()
        
        self.input_queue.close()
        self.output_queue.close()
    
    def init_timer(self):
        self.start_time = time.time()
        self.generated_size = 0

    @staticmethod
    def _gen_per_process(
        latent_model_path,
        dtype,
        rank, 
        input_queue,
        output_queue,
        max_length,
        max_new_tokens,
        topk_interpolation,
        add_gumbel_noise,
        gumbel_temperature,
        noise_scale,
    ):
        device = torch.device(f'cuda:{rank}')
        
        model = AutoModelForCausalLM.from_pretrained(latent_model_path,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                use_cache=False,
                trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(latent_model_path)
        
        model.eval()
        model.tokenizer = tokenizer
        
        model.latent_token_ids = tokenizer(['<think>','</think>'], add_special_tokens=False)['input_ids']
        
        while True:
            batch_id, batch_examples = input_queue.get()
            
            batch_results = []
            for examples in batch_examples:
                text_input = prepare_single_example(examples, model, latent_model_path)

                decoded_output = LatentSFTStage2SoftEmbedding.one_example_generate_hf(
                    model,
                    text_input,
                    max_new_tokens=max_new_tokens, 
                    temperature=0.6,               
                    top_p=0.95,
                    do_sample=True,
                    topk_interpolation=topk_interpolation,
                    # Gumbel noise
                    add_gumbel_noise=add_gumbel_noise,
                    gumbel_temperature=gumbel_temperature,
                    noise_scale=noise_scale,
                )

                decoded_text = decoded_output['text'] 
                decoded = (decoded_text, decoded_output["generate_token_num"])
                batch_results.append(decoded)
            
            output_queue.put((batch_id, batch_results))

    def _gen(
        self,
        text_input,
        show_progress_bar: bool = False
    ):
        batch_size = self.batch_size
        n = len(text_input)
        
        # Sort by problem length to balance load across batches
        sorted_indices = sorted(range(n), key=lambda i: len(text_input[i].get('problem', '')))
        sorted_input = [text_input[i] for i in sorted_indices]
        
        num_batches = 0
        for start in range(0, n, batch_size):
            batch = sorted_input[start: start + batch_size]
            self.input_queue.put((start, batch))
            num_batches += 1

        if show_progress_bar:
            pbar = tqdm(total=n, desc=f'Generate size: {self.generated_size}, consumed time: {round(time.time() - self.start_time, 2)}s')
        
        id_gen_texts = []
        for _ in range(num_batches):
            batch_id, gen_texts = self.output_queue.get()
            id_gen_texts.append((batch_id, gen_texts))
            if show_progress_bar:
                pbar.update(len(gen_texts))
        if show_progress_bar:
            pbar.close()
        
        # Flatten in sorted order
        id_gen_texts.sort(key=lambda x: x[0])
        sorted_texts = []
        for _, texts in id_gen_texts:
            sorted_texts.extend(texts)
        
        # Restore original order
        original_order_texts = [None] * n
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            original_order_texts[orig_idx] = sorted_texts[sorted_pos]
        
        self.generated_size += n
        return original_order_texts

    def gen(
        self, 
        text_input,
        show_progress_bar=True,
        **kwargs
    ):

        gen_texts = self._gen(text_input, show_progress_bar)
        
        return gen_texts
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data    
def write_jsonl(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GSM8k', type=str, choices=['GSM8k', 'Math500', 'AIME24'])
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the evaluation jsonl file for the selected --dataset')
    parser.add_argument('--check_point', type=str, required=True,
                        help='Stage-2 latent-model checkpoint step to evaluate, e.g. "377"')
    parser.add_argument('--latent_model_path', type=str, required=True,
                        help='Base directory of the stage-2 latent-model run; the '
                             'actual model is loaded from '
                             '<latent_model_path>/checkpoint-<check_point>/hf')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to write evaluation results to')
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--topk_interpolation', type=int, default=10)
    # Gumbel noise args
    parser.add_argument('--add_gumbel_noise', type=bool, default=False)
    parser.add_argument('--gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--noise_scale', type=float, default=0.5)
    
    args = parser.parse_args()
    check_point = args.check_point
    args.latent_model_path = os.path.join(args.latent_model_path, f'checkpoint-{check_point}/hf')

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")


    model = MultiprocessTransformerWrapper(
        latent_model_path=args.latent_model_path,
        dtype=args.dtype,
        mp_size=args.mp_size,
        max_length=args.max_length,
        max_new_tokens = args.max_new_tokens,
        topk_interpolation = args.topk_interpolation,
        add_gumbel_noise = args.add_gumbel_noise,
        gumbel_temperature = args.gumbel_temperature,
        noise_scale = args.noise_scale,
        batch_size = args.batch_size,
    )
    data = read_jsonl(args.data_path)

    gen_texts = model.gen(data)
    for i in range(len(data)):
        data[i]['prediction'] = gen_texts[i][0]

    token_nums = [i[1] for i in gen_texts]
    avg_token_num = sum(token_nums) / len(token_nums)

    model.close()
    
    # print(gen_texts[:1])
    results = []
    ## evaluate
    for i in range(len(data)):
        pred = extract_answer(gen_texts[i][0])
        results.append(check_is_correct(pred, data[i]["answer"]))
    print('=======================================================')
    print(f'{args.dataset} Scores: {sum(results)/len(results)}  AVG Tokens: {avg_token_num}')



    # print(data[:10])
    os.makedirs(args.save_path, exist_ok=True)
    # Append gumbel info to filename if enabled
    gumbel_suffix = ""
    if args.add_gumbel_noise:
        gumbel_suffix = f"_gumbel_{args.gumbel_temperature}_{args.noise_scale}"
    write_jsonl(data, os.path.join(args.save_path, f'{check_point}_{args.max_new_tokens}_{args.dataset}{gumbel_suffix}_result.jsonl'))
    eval_txt_path = os.path.join(args.save_path, f'eval_result_{args.dataset}.txt')
    with open(eval_txt_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('=======================================================\n')
        f.write(f'{args.dataset} Scores: {sum(results)/len(results)}   AVG Tokens: {avg_token_num}  check_point: {check_point} mex_new_tokens: {args.max_new_tokens} gumbel: {args.add_gumbel_noise} temp: {args.gumbel_temperature} scale: {args.noise_scale}'+'\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
