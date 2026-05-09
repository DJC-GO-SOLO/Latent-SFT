import os
import sys
import json
import math
import time
import signal
import socket
import argparse
from typing import Any
import torch.multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


from transformers import AutoTokenizer




# =========================
# Utils (math scoring functions below are adapted from the verl library:
# https://github.com/volcengine/verl)
# =========================


def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

# =========================
# End of verl-adapted scoring utilities
# =========================

def read_jsonl(file_path):
    data: list[Any] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(data, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def append_jsonl(item, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "a", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")


def get_last_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenization produced empty ids for marker: {text!r}")
    return ids[0]


def build_prompt_text_qwen(item, tokenizer):
    """Qwen chat template prompt. Ensures the prompt ends with ``<think>``."""
    problem = item.get("problem", None) or item.get("question", "")
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Append the ``<think>`` marker only if the chat template did not already
    # include it, so we don't duplicate the token across different templates.
    if not prompt.rstrip().endswith("<think>"):
        prompt = prompt + "<think>"
    return prompt


def batched_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def kill_process_tree(pid):
    """Forcefully kill a process together with all of its descendants."""
    try:
        import subprocess
        # Find all direct child processes.
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        )
        for child_pid in result.stdout.strip().split("\n"):
            if child_pid:
                kill_process_tree(int(child_pid))  # Recursively kill children.
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, ValueError, OSError):
        pass


def is_port_free(port):
    """Check whether a local TCP port is currently free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def wait_for_ports_free(gpu_ids, timeout=60):
    """Wait until every MASTER_PORT bound to the given GPUs has been released."""
    ports = [29500 + int(gid) for gid in gpu_ids]
    start = time.time()
    while time.time() - start < timeout:
        all_free = all(is_port_free(p) for p in ports)
        if all_free:
            return True
        time.sleep(2)
    print(f"  Warning: Ports {ports} not fully released after {timeout}s, proceeding anyway...")
    return False

# =========================
# DP Worker Function
# =========================
def inference_worker(gpu_id, prompt_chunk, result_queue, config):
    """Inference worker that runs inside an isolated subprocess."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + int(gpu_id))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
        latent_end_token_id = get_last_token_id(tokenizer, "</think>")

        llm = sgl.Engine(
            model_path=config["model_path"],
            trust_remote_code=True,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            tp_size=1,
            enable_latent=True,
            latent_end_token_id=latent_end_token_id,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            mem_fraction_static=0.90,
            sampling_backend="flashinfer",
            max_running_requests=2048,
            log_level="error",
            skip_tokenizer_init=True,
            max_topk=config["max_topk"],
        )

        local_predictions = []
        batch_size = config["gen_batch_size"]

        for batch_prompts in batched_list(prompt_chunk, batch_size):
            outputs = llm.generate(
                input_ids=batch_prompts,
                sampling_params={
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_new_tokens": config["max_new_tokens"],
                    "gumbel_softmax_temperature": config["gumbel_softmax_temperature"],
                    "noise_scale": config["noise_scale"],
                    "add_noise_gumbel_softmax": config["add_noise_gumbel_softmax"],
                },
                return_logprob=True,
            )

            batch_output_ids = [o['output_ids'] for o in outputs]
            batch_output_lens = [len(ids) for ids in batch_output_ids]
            decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)
            local_predictions.extend(list(zip(decoded_texts, batch_output_lens)))

        llm.shutdown()

        temp_file = os.path.join(config["output_dir"], f"temp_preds_gpu_{gpu_id}.json")
        os.makedirs(config["output_dir"], exist_ok=True)
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(local_predictions, f)  # list of (text, length) tuples

        result_queue.put(temp_file)

    except Exception as e:
        print(f"Worker {gpu_id} failed: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(None)


# =========================
# Single Checkpoint Eval
# =========================
def eval_single_checkpoint(model_path, data, config, target_gpu_ids):
    """Evaluate a single checkpoint and return its accuracy and statistics."""
    config = config.copy()
    config["model_path"] = model_path

    total_len = len(data)

    # Tokenize prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text_prompts = [build_prompt_text_qwen(item, tokenizer) for item in data]

    encodings = tokenizer(
        text_prompts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    input_ids_all = encodings["input_ids"]

    # Split across GPUs
    num_gpus = len(target_gpu_ids)
    chunk_size = math.ceil(total_len / num_gpus)
    prompt_chunks = [input_ids_all[i : i + chunk_size] for i in range(0, total_len, chunk_size)]

    # Launch workers
    result_queue = mp.Queue()
    processes = []

    for i, physical_gpu_id in enumerate(target_gpu_ids):
        if i >= len(prompt_chunks):
            break
        p = mp.Process(
            target=inference_worker,
            args=(physical_gpu_id, prompt_chunks[i], result_queue, config)
        )
        p.start()
        processes.append(p)

    # Collect results
    temp_files = []
    for _ in range(len(processes)):
        res = result_queue.get()
        if res:
            temp_files.append(res)

    for p in processes:
        # p.join()
        p.join(timeout=30)

    # Force kill any lingering child processes (sglang spawns subprocesses)
    for p in processes:
        if p.is_alive():
            kill_process_tree(p.pid)
            p.join(timeout=5)
        elif p.pid:
            kill_process_tree(p.pid)

    # Merge results (sort by GPU id to preserve order)
    temp_files.sort(key=lambda x: int(x.split("_gpu_")[-1].replace(".json", "")))

    predictions = []
    output_lengths = []
    for fpath in temp_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            chunk_preds = json.load(f)  # list of [text, length] pairs
            for text, length in chunk_preds:
                predictions.append(text)
                output_lengths.append(length)
        os.remove(fpath)

    if len(predictions) != len(data):
        print(f"  Warning: Predictions({len(predictions)}) != Data({len(data)})")

    # Score
    def score_chunk_fn(indices_range):
        start, end = indices_range
        chunk_scores = []
        for i in range(start, end):
            if i >= len(predictions):
                break
            pred = predictions[i]
            ans = data[i].get("answer", data[i].get("solution", ""))
            try:
                score = compute_score(pred, ans)
            except Exception:
                score = 0.0
            chunk_scores.append(score)
        return chunk_scores

    score_batch_size = 10000
    chunk_ranges = [
        (i, min(i + score_batch_size, len(predictions)))
        for i in range(0, len(predictions), score_batch_size)
    ]

    scores = []
    with ThreadPoolExecutor(max_workers=config["num_score_workers"]) as ex:
        for chunk_res in ex.map(score_chunk_fn, chunk_ranges):
            scores.extend(chunk_res)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    avg_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0.0
    return accuracy, len(scores), avg_len, predictions, scores, output_lengths


# =========================
# Main
# =========================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Evaluate Math500 across multiple checkpoints")
    parser.add_argument("--base_model_dir", type=str, required=True,
                        help="Base directory containing checkpoint folders, "
                             "e.g. /path/to/stage2results/<run_name>")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the Math500 test jsonl file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path for result jsonl. Default: {base_model_dir}/math500_eval_results.jsonl")
    parser.add_argument("--ckpt_start", type=int, default=450, help="First checkpoint number")
    parser.add_argument("--ckpt_step", type=int, default=450, help="Step between checkpoints")
    parser.add_argument("--ckpt_count", type=int, default=50, help="Number of checkpoints to evaluate")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated physical GPU IDs to use")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_topk", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=1024)
    parser.add_argument("--num_score_workers", type=int, default=32)
    parser.add_argument("--gumbel_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--add_noise_gumbel_softmax", default=False)
    parser.add_argument("--save_predictions", default=False,
                        help="Save per-checkpoint generation results to jsonl")
    args = parser.parse_args()

    target_gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    output_path = args.output_path or os.path.join(args.base_model_dir, "math500_eval_results.jsonl")

    config = {
        "output_dir": os.path.join(args.base_model_dir, "_temp_eval"),
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_topk": args.max_topk,
        "gen_batch_size": args.gen_batch_size,
        "num_score_workers": args.num_score_workers,
    }

    # Read data once
    data = read_jsonl(args.data_path)
    print(f"Data: {args.data_path} ({len(data)} samples)")
    print(f"GPUs: {target_gpu_ids}")
    print(f"Results will be saved to: {output_path}")

    # Generate checkpoint list
    checkpoints = [args.ckpt_start + i * args.ckpt_step for i in range(args.ckpt_count)]

    print(f"\n{'='*60}")
    print(f"Evaluating {len(checkpoints)} checkpoints: {checkpoints[0]} → {checkpoints[-1]} (step={args.ckpt_step})")
    print(f"{'='*60}\n")

    pbar = tqdm(checkpoints, desc="Checkpoints")
    for ckpt in pbar:
        model_path = os.path.join(args.base_model_dir, f"checkpoint-{ckpt}", "hf")

        if not os.path.exists(model_path):
            pbar.write(f"  [SKIP] {model_path} not found")
            continue

        pbar.set_description(f"Checkpoint-{ckpt}")

        accuracy, total, avg_len, predictions, scores, output_lengths = eval_single_checkpoint(model_path, data, config, target_gpu_ids)

        result = {
            "checkpoint": ckpt,
            "model_path": model_path,
            "accuracy": round(accuracy, 6),
            "total_samples": total,
            "correct": int(round(accuracy * total)),
            "avg_output_length": round(avg_len, 1),
            # generation params
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_topk": args.max_topk,
            "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
            "noise_scale": args.noise_scale,
            "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
            "gen_batch_size": args.gen_batch_size,
        }
        append_jsonl(result, output_path)

        # Save per-sample predictions
        if args.save_predictions:
            pred_dir = os.path.join(args.base_model_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            pred_file = os.path.join(pred_dir, f"checkpoint-{ckpt}.jsonl")
            with open(pred_file, "w", encoding="utf-8") as f:
                for idx in range(min(len(data), len(predictions))):
                    item = {
                        "problem": data[idx].get("problem", data[idx].get("question", "")),
                        "answer": data[idx].get("answer", data[idx].get("solution", "")),
                        "prediction": predictions[idx],
                        "score": scores[idx] if idx < len(scores) else None,
                        "output_token_length": output_lengths[idx] if idx < len(output_lengths) else None,
                    }
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

        pbar.write(f"  checkpoint-{ckpt}: {result['correct']}/{total} = {accuracy:.4f} | avg_len={avg_len:.1f}")
        # Wait for sglang engine ports to fully release before next checkpoint
        wait_for_ports_free(target_gpu_ids, timeout=60)
        

    # Final summary
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_path}")
    if os.path.exists(output_path):
        results = read_jsonl(output_path)
        best = max(results, key=lambda x: x["accuracy"])
        print(f"Best: checkpoint-{best['checkpoint']} = {best['accuracy']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
