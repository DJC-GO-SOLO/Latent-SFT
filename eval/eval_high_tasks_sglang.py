import os
import re
import json
import math
import time
import random
import signal
import socket
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer


GPQA_OPTION_LETTERS = ["A", "B", "C", "D"]


PASS_K_VALUES = [1, 4, 8, 16, 32, 64]


# =========================
# Math scoring helpers (inlined from math500 eval script)
# =========================


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
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


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
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]


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
    retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]
    return retval


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


def normalize_integer_answer(text):
    normalized = strip_string(str(text).strip())
    if not re.fullmatch(r"[+-]?\d+", normalized):
        return None
    return str(int(normalized))


def compute_aime_score(solution_str, ground_truth):
    score = compute_score(solution_str, ground_truth)
    if score == 1.0:
        return 1.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is None:
            return 0.0
        answer = remove_boxed(string_in_last_boxed)
    except Exception:
        return 0.0

    normalized_answer = normalize_integer_answer(answer)
    normalized_ground_truth = normalize_integer_answer(ground_truth)
    if normalized_answer is None or normalized_ground_truth is None:
        return 0.0
    return 1.0 if normalized_answer == normalized_ground_truth else 0.0


# =========================
# Prompt / tokenizer helpers
# =========================


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
GPQA_USER_SUFFIX = "\nPlease provide the option letter (A, B, C, or D) as your final answer."


def build_prompt_text_qwen(item, tokenizer, dataset_name=None):
    """Qwen chat template prompt"""
    problem = item.get("problem", None) or item.get("question", "")
    if dataset_name == "gpqa":
        problem = problem + GPQA_USER_SUFFIX
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt# + "<think>"


def get_last_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenization produced empty ids for marker: {text!r}")
    return ids[0]


# =========================
# Process / port management helpers
# =========================


def kill_process_tree(pid):
    """Force kill a process and all its children"""
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        )
        for child_pid in result.stdout.strip().split("\n"):
            if child_pid:
                kill_process_tree(int(child_pid))
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, ValueError, OSError):
        pass


def is_port_free(port):
    """Check if a port is free"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def wait_for_ports_free(gpu_ids, timeout=60):
    """Wait for all GPU MASTER_PORTs to be released"""
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
# Data I/O
# =========================


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line at {file_path}:{line_idx}: {exc}") from exc
    return data


def write_json(path, data):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_problem_text(item, dataset_name, sample_idx, data_path):
    problem = item.get("problem")
    if problem is None:
        problem = item.get("question")
    if problem is None:
        raise ValueError(
            f"Missing problem/question field in dataset={dataset_name}, sample_idx={sample_idx}, path={data_path}"
        )
    problem = str(problem).strip()
    if not problem:
        raise ValueError(
            f"Empty problem/question field in dataset={dataset_name}, sample_idx={sample_idx}, path={data_path}"
        )
    return problem


def get_answer_text(item, dataset_name, sample_idx, data_path):
    answer = item.get("answer")
    if answer is None:
        answer = item.get("solution")
    if answer is None:
        raise ValueError(
            f"Missing answer/solution field in dataset={dataset_name}, sample_idx={sample_idx}, path={data_path}"
        )
    answer = str(answer).strip()
    if not answer:
        raise ValueError(
            f"Empty answer/solution field in dataset={dataset_name}, sample_idx={sample_idx}, path={data_path}"
        )
    return answer


def load_gpqa_records(data_path):
    """Load GPQA Diamond dataset in multiple-choice format.
    
    Each record gets a shuffled set of 4 options (A/B/C/D) and the correct
    option letter is stored as the answer.
    """
    raw = read_jsonl(data_path)
    if not raw:
        raise ValueError(f"Dataset is empty: gpqa ({data_path})")
    records = []
    for idx, item in enumerate(raw):
        question = item.get("Question", "").strip()
        if not question:
            raise ValueError(f"Missing Question field in gpqa, sample_idx={idx}, path={data_path}")
        correct = item.get("Correct Answer", "").strip()
        incorrect1 = item.get("Incorrect Answer 1", "").strip()
        incorrect2 = item.get("Incorrect Answer 2", "").strip()
        incorrect3 = item.get("Incorrect Answer 3", "").strip()
        if not correct:
            raise ValueError(f"Missing Correct Answer in gpqa, sample_idx={idx}, path={data_path}")

        # Build options and shuffle with a fixed seed per sample for reproducibility
        options = [
            (correct, True),
            (incorrect1, False),
            (incorrect2, False),
            (incorrect3, False),
        ]
        rng = random.Random(42 + idx)
        rng.shuffle(options)

        correct_letter = None
        option_lines = []
        for i, (text, is_correct) in enumerate(options):
            letter = GPQA_OPTION_LETTERS[i]
            option_lines.append(f"({letter}) {text}")
            if is_correct:
                correct_letter = letter

        # Build the full problem text with options
        problem_with_options = question + "\n" + "\n".join(option_lines)
        records.append({
            "problem": problem_with_options,
            "answer": correct_letter,  # e.g., "A", "B", "C", or "D"
        })
    return records


def load_dataset_records(dataset_name, data_path):
    if dataset_name == "gpqa":
        return load_gpqa_records(data_path)
    raw = read_jsonl(data_path)
    if not raw:
        raise ValueError(f"Dataset is empty: {dataset_name} ({data_path})")
    records = []
    for idx, item in enumerate(raw):
        records.append(
            {
                "problem": get_problem_text(item, dataset_name, idx, data_path),
                "answer": get_answer_text(item, dataset_name, idx, data_path),
            }
        )
    return records


# =========================
# GPQA scoring (multiple-choice, LightEval style)
# =========================


def extract_gpqa_option(solution_str):
    """Extract the option letter (A/B/C/D) from the model's \\boxed{} output."""
    # Try to extract from \boxed{}
    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            content = remove_boxed(boxed).strip()
        except Exception:
            content = ""
        # Match single letter A/B/C/D (possibly with parentheses, period, etc.)
        m = re.match(r"^\(?([A-Da-d])\)?\.?$", content)
        if m:
            return m.group(1).upper()
        # If content is exactly one of the option letters
        if content.upper() in GPQA_OPTION_LETTERS:
            return content.upper()
    # Fallback: search for the last occurrence of a standalone option letter in the text
    matches = re.findall(r"\b([A-Da-d])\b", solution_str)
    if matches:
        return matches[-1].upper()
    return None


def compute_gpqa_score(solution_str, ground_truth):
    """Score GPQA by matching extracted option letter against the correct option letter."""
    predicted = extract_gpqa_option(solution_str)
    if predicted is None:
        return 0.0
    # ground_truth is the correct option letter (e.g., "A", "B", "C", "D")
    return 1.0 if predicted == ground_truth.strip().upper() else 0.0

# =========================
# Batch / inference utilities
# =========================


def batched_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def persistent_inference_worker(gpu_id, work_queue, result_queue, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + int(gpu_id))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    import torch
    import sglang as sgl

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

        result_queue.put(("ready", gpu_id))

        while True:
            item = work_queue.get()
            if item is None:
                break

            prompt_chunk, temp_file_path, run_seed = item
            seed_value = int(run_seed) + int(gpu_id)
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

            local_predictions = []
            for batch_prompts in batched_list(prompt_chunk, config["gen_batch_size"]):
                outputs = llm.generate(
                    input_ids=batch_prompts,
                    sampling_params={
                        "temperature": config["temperature"],
                        "top_p": config["top_p"],
                        "max_new_tokens": config["max_new_tokens"],
                        "gumbel_softmax_temperature": config["gumbel_softmax_temperature"],
                        "noise_scale": config["noise_scale"],
                        "add_noise_gumbel_softmax": config["add_noise_gumbel_softmax"],
                        "use_one_sided_gumbel_noise": config["use_one_sided_gumbel_noise"],
                    },
                    return_logprob=True,
                )
                batch_output_ids = [o["output_ids"] for o in outputs]
                batch_output_lens = [len(ids) for ids in batch_output_ids]
                decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)
                local_predictions.extend(list(zip(decoded_texts, batch_output_lens)))

            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(local_predictions, f)

            result_queue.put(("done", temp_file_path))

        llm.shutdown()

    except Exception as e:
        print(f"Worker {gpu_id} failed: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put(("error", str(e)))


def launch_persistent_workers(target_gpu_ids, config):
    work_queues = {gpu_id: mp.Queue() for gpu_id in target_gpu_ids}
    result_queue = mp.Queue()
    processes = []

    for gpu_id in target_gpu_ids:
        p = mp.Process(target=persistent_inference_worker, args=(gpu_id, work_queues[gpu_id], result_queue, config))
        p.start()
        processes.append((gpu_id, p))

    ready_count = 0
    while ready_count < len(target_gpu_ids):
        msg_type, payload = result_queue.get()
        if msg_type == "ready":
            print(f"  Worker GPU {payload} ready.")
            ready_count += 1
        elif msg_type == "error":
            raise RuntimeError(f"Worker failed during init: {payload}")
        else:
            raise RuntimeError(f"Unexpected worker init message: {msg_type}")

    return processes, work_queues, result_queue


def shutdown_workers(processes, work_queues):
    for gpu_id, _ in processes:
        work_queues[gpu_id].put(None)
    for _, p in processes:
        p.join(timeout=60)
        if p.is_alive():
            kill_process_tree(p.pid)
            p.join(timeout=5)
        elif p.pid:
            kill_process_tree(p.pid)


def build_input_ids(records, tokenizer, dataset_name=None):
    prompts = [build_prompt_text_qwen({"problem": item["problem"]}, tokenizer, dataset_name=dataset_name) for item in records]
    encodings = tokenizer(
        prompts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )
    input_ids_all = encodings["input_ids"]
    if len(input_ids_all) != len(records):
        raise ValueError("Tokenized prompt count does not match dataset size.")
    return input_ids_all


def score_prediction(dataset_name, prediction, answer):
    if dataset_name in {"aime24", "aime25"}:
        return compute_aime_score(prediction, answer)
    if dataset_name == "gpqa":
        return compute_gpqa_score(prediction, answer)
    return compute_score(prediction, answer)


def run_one_eval(dataset_name, records, input_ids_all, config, target_gpu_ids, work_queues, result_queue, run_tag, run_seed):
    total_len = len(records)
    chunk_size = math.ceil(total_len / len(target_gpu_ids))
    prompt_chunks = [input_ids_all[i : i + chunk_size] for i in range(0, total_len, chunk_size)]

    active_gpu_ids = []
    for i, gpu_id in enumerate(target_gpu_ids):
        if i >= len(prompt_chunks):
            break
        temp_file = os.path.join(config["output_dir"], f"temp_preds_gpu_{gpu_id}_{run_tag}.json")
        work_queues[gpu_id].put((prompt_chunks[i], temp_file, run_seed))
        active_gpu_ids.append(gpu_id)

    temp_files = []
    for _ in range(len(active_gpu_ids)):
        msg_type, payload = result_queue.get()
        if msg_type == "done":
            temp_files.append(payload)
        elif msg_type == "error":
            raise RuntimeError(f"Worker error during inference: {payload}")
        else:
            raise RuntimeError(f"Unexpected worker message during inference: {msg_type}")

    temp_files.sort(key=lambda x: int(x.split("_gpu_")[1].split("_")[0]))

    predictions = []
    output_lengths = []
    for fpath in temp_files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunk_preds = json.load(f)
            for text, length in chunk_preds:
                predictions.append(text)
                output_lengths.append(length)
        os.remove(fpath)

    if len(predictions) != total_len:
        raise ValueError(f"Predictions({len(predictions)}) != Data({total_len}) for dataset={dataset_name}")

    def score_chunk_fn(indices_range):
        start, end = indices_range
        chunk_scores = []
        for i in range(start, end):
            try:
                score = score_prediction(dataset_name, predictions[i], records[i]["answer"])
            except Exception:
                score = 0.0
            chunk_scores.append(score)
        return chunk_scores

    chunk_ranges = [(i, min(i + 10000, total_len)) for i in range(0, total_len, 10000)]
    scores = []
    with ThreadPoolExecutor(max_workers=config["num_score_workers"]) as ex:
        for chunk_res in ex.map(score_chunk_fn, chunk_ranges):
            scores.extend(chunk_res)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    avg_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0.0
    return accuracy, len(scores), avg_len, scores, predictions


# =========================
# Pass@k computation
# =========================


def estimate_pass_at_k(num_samples, num_correct, k):
    if num_samples < k:
        return None
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def compute_dataset_pass_at_k(dataset_result):
    runs = sorted(dataset_result["per_run"], key=lambda x: x["run_index"])
    num_runs = len(runs)
    num_samples = dataset_result["num_samples"]
    if num_runs == 0:
        return {f"pass@{k}": None for k in PASS_K_VALUES}

    correct_counts = [0] * num_samples
    for run in runs:
        for idx, score in enumerate(run["sample_correct"]):
            correct_counts[idx] += int(score)

    result = {}
    for k in PASS_K_VALUES:
        if k > num_runs:
            result[f"pass@{k}"] = None
            continue
        values = [estimate_pass_at_k(num_runs, c, k) for c in correct_counts]
        result[f"pass@{k}"] = round(sum(values) / len(values), 6) if values else 0.0
    return result


def compute_dataset_summary(dataset_result):
    runs = sorted(dataset_result["per_run"], key=lambda x: x["run_index"])
    dataset_result["completed_num_runs"] = len(runs)
    dataset_result["mean_accuracy"] = round(sum(x["accuracy"] for x in runs) / len(runs), 6) if runs else None
    dataset_result["mean_output_length"] = round(sum(x["avg_output_length"] for x in runs) / len(runs), 1) if runs else None
    dataset_result["pass@k"] = compute_dataset_pass_at_k(dataset_result)


def compute_macro_pass_at_k(all_results):
    macro = {}
    for k in PASS_K_VALUES:
        values = []
        key = f"pass@{k}"
        for dataset_result in all_results.values():
            value = dataset_result.get("pass@k", {}).get(key)
            if value is not None:
                values.append(value)
        macro[key] = round(sum(values) / len(values), 6) if values else None
    return macro


# =========================
# Result validation / persistence
# =========================


def validate_run(run_result, expected_num_samples):
    required_keys = ["run_index", "seed", "accuracy", "correct", "total_samples", "avg_output_length", "sample_correct"]
    if any(key not in run_result for key in required_keys):
        return False
    if run_result["total_samples"] != expected_num_samples:
        return False
    if len(run_result["sample_correct"]) != expected_num_samples:
        return False
    return True


def load_existing_output(output_path):
    if not os.path.exists(output_path):
        return None
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_run_seed(base_seed, dataset_index, run_index):
    return int(base_seed) + dataset_index * 100000 + run_index


def format_metric(value):
    return "NA" if value is None else f"{value:.4f}"


def save_results(output_path, model_path, datasets_config, all_results, args):
    """Save detailed results including per_run data (for checkpoint / resume)."""
    final_result = {
        "model_path": model_path,
        "datasets_config": datasets_config,
        "datasets": all_results,
        "macro_average_pass@k": compute_macro_pass_at_k(all_results),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_topk": args.max_topk,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "use_one_sided_gumbel_noise": args.use_one_sided_gumbel_noise,
        "gen_batch_size": args.gen_batch_size,
        "num_score_workers": args.num_score_workers,
        "base_seed": args.base_seed,
        "pass_k_values": PASS_K_VALUES,
    }
    write_json(output_path, final_result)

def save_summary(output_path, model_path, all_results, args):
    """Save a lightweight summary with only pass@k per dataset."""
    summary_path = output_path.replace(".json", "_summary.json")
    datasets_summary = {}
    for ds_name, ds_result in all_results.items():
        datasets_summary[ds_name] = {
            "dataset_name": ds_result.get("dataset_name", ds_name),
            "num_samples": ds_result.get("num_samples"),
            "completed_num_runs": ds_result.get("completed_num_runs"),
            "mean_accuracy": ds_result.get("mean_accuracy"),
            "mean_output_length": ds_result.get("mean_output_length"),
            "pass@k": ds_result.get("pass@k", {}),
        }
    summary = {
        "model_path": model_path,
        "datasets": datasets_summary,
        "macro_average_pass@k": compute_macro_pass_at_k(all_results),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "use_one_sided_gumbel_noise": args.use_one_sided_gumbel_noise,
    }
    write_json(summary_path, summary)


# =========================
# Main
# =========================


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Evaluate high tasks with the original Math500 SGLang template and sampling config")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--math500_path", type=str, default="Math-500-test.jsonl")
    parser.add_argument("--aime24_path", type=str, default="AIME-2024-test.jsonl")
    parser.add_argument("--aime25_path", type=str, default="AIME-2025-test.jsonl")
    parser.add_argument("--gpqa_path", type=str, default="gpqa_diamond.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_topk", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=1024)
    parser.add_argument("--num_score_workers", type=int, default=32)
    parser.add_argument("--gumbel_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--add_noise_gumbel_softmax", default=False)
    parser.add_argument(
        "--use_one_sided_gumbel_noise",
        action="store_true",
        default=False,
        help="Enable one-sided Gumbel noise sampling.",
    )
    parser.add_argument("--base_seed", type=int, default=12345)
    parser.add_argument("--save_predictions", action="store_true", default=True, help="Save model predictions to a separate JSONL file per dataset per run")
    args = parser.parse_args()

    target_gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if not target_gpu_ids:
        raise ValueError("gpu_ids must not be empty.")

    # Encode sampling params into filenames and temporary directories to avoid
    # overwriting across configs.
    sampling_tag = (
        f"gumbel{args.gumbel_softmax_temperature}"
        f"_noise{args.noise_scale}"
        f"_addnoise{args.add_noise_gumbel_softmax}"
        f"_onesided{args.use_one_sided_gumbel_noise}"
    )
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.join(args.model_path, f"high_tasks_eval_result_new_{sampling_tag}.json")
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else args.model_path

    config = {
        "model_path": args.model_path,
        "output_dir": os.path.join(output_dir, f"_temp_eval_{sampling_tag}"),
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "noise_scale": args.noise_scale,
        "add_noise_gumbel_softmax": args.add_noise_gumbel_softmax,
        "use_one_sided_gumbel_noise": args.use_one_sided_gumbel_noise,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_topk": args.max_topk,
        "gen_batch_size": args.gen_batch_size,
        "num_score_workers": args.num_score_workers,
    }

    datasets_config = [
        {"dataset_name": "math500", "data_path": args.math500_path, "num_runs": 4},
        {"dataset_name": "aime24", "data_path": args.aime24_path, "num_runs": 64},
        {"dataset_name": "aime25", "data_path": args.aime25_path, "num_runs": 64},
        {"dataset_name": "gpqa", "data_path": args.gpqa_path, "num_runs": 8},
    ]

    print(f"{'=' * 72}")
    print(f"Model: {args.model_path}")
    print(f"GPUs:  {target_gpu_ids}")
    print(f"Output: {output_path}")
    print(f"{'=' * 72}\n")

    existing_output = load_existing_output(output_path)
    all_results = existing_output.get("datasets", {}) if existing_output else {}

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    dataset_bundles = []
    for dataset_cfg in datasets_config:
        dataset_name = dataset_cfg["dataset_name"]
        data_path = dataset_cfg["data_path"]
        records = load_dataset_records(dataset_name, data_path)
        input_ids_all = build_input_ids(records, tokenizer, dataset_name=dataset_name)

        old_dataset = all_results.get(dataset_name, {})
        cleaned_runs = []
        for run_result in sorted(old_dataset.get("per_run", []), key=lambda x: x.get("run_index", -1)):
            if validate_run(run_result, len(records)):
                cleaned_runs.append(run_result)
            else:
                print(f"Filtered incomplete stored run: dataset={dataset_name}, run={run_result.get('run_index')}")

        all_results[dataset_name] = {
            "dataset_name": dataset_name,
            "data_path": data_path,
            "num_samples": len(records),
            "target_num_runs": dataset_cfg["num_runs"],
            "per_run": cleaned_runs,
        }
        compute_dataset_summary(all_results[dataset_name])

        dataset_bundles.append(
            {
                "dataset_name": dataset_name,
                "num_runs": dataset_cfg["num_runs"],
                "records": records,
                "input_ids_all": input_ids_all,
            }
        )

    save_results(output_path, args.model_path, datasets_config, all_results, args)
    save_summary(output_path, args.model_path, all_results, args)

    print("Launching persistent workers (loading model)...")
    processes, work_queues, result_queue = launch_persistent_workers(target_gpu_ids, config)
    print(f"All {len(target_gpu_ids)} workers ready.\n")

    try:
        for dataset_index, bundle in enumerate(dataset_bundles):
            dataset_name = bundle["dataset_name"]
            dataset_result = all_results[dataset_name]
            completed = {x["run_index"] for x in dataset_result["per_run"]}

            print(f"\n{'=' * 72}")
            print(f"Dataset: {dataset_name} ({dataset_result['num_samples']} samples) | target_runs={bundle['num_runs']}")
            print(f"{'=' * 72}")

            for run_index in tqdm(range(1, bundle["num_runs"] + 1), desc=f"{dataset_name} runs", leave=False):
                if run_index in completed:
                    continue

                config["output_dir"] = os.path.join(output_dir, f"_temp_eval_{sampling_tag}", dataset_name, f"run{run_index}")
                run_seed = make_run_seed(args.base_seed, dataset_index, run_index)
                run_tag = f"{dataset_name}_run{run_index}"

                accuracy, total, avg_len, scores, preds = run_one_eval(
                    dataset_name=dataset_name,
                    records=bundle["records"],
                    input_ids_all=bundle["input_ids_all"],
                    config=config,
                    target_gpu_ids=target_gpu_ids,
                    work_queues=work_queues,
                    result_queue=result_queue,
                    run_tag=run_tag,
                    run_seed=run_seed,
                )

                if args.save_predictions:
                    pred_dir = os.path.join(output_dir, f"predictions_{sampling_tag}", dataset_name)
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_file = os.path.join(pred_dir, f"run{run_index}.jsonl")
                    with open(pred_file, "w", encoding="utf-8") as pf:
                        for sidx, (pred_text, score_val) in enumerate(zip(preds, scores)):
                            pf.write(json.dumps({
                                "index": sidx,
                                "problem": bundle["records"][sidx]["problem"],
                                "answer": bundle["records"][sidx]["answer"],
                                "prediction": pred_text,
                                "score": int(score_val),
                            }, ensure_ascii=False) + "\n")
                    tqdm.write(f"    Predictions saved to {pred_file}")

                dataset_result["per_run"].append(
                    {
                        "run_index": run_index,
                        "seed": run_seed,
                        "accuracy": round(accuracy, 6),
                        "correct": int(round(accuracy * total)),
                        "total_samples": total,
                        "avg_output_length": round(avg_len, 1),
                        "sample_correct": [int(x) for x in scores],
                    }
                )
                dataset_result["per_run"] = sorted(dataset_result["per_run"], key=lambda x: x["run_index"])
                compute_dataset_summary(dataset_result)
                save_results(output_path, args.model_path, datasets_config, all_results, args)
                save_summary(output_path, args.model_path, all_results, args)

                metrics = dataset_result["pass@k"]
                tqdm.write(
                    f"  {dataset_name} run {run_index}/{bundle['num_runs']}: "
                    f"acc={accuracy:.4f} | avg_len={avg_len:.1f} | "
                    f"pass@1={format_metric(metrics['pass@1'])} | "
                    f"pass@4={format_metric(metrics['pass@4'])} | "
                    f"pass@8={format_metric(metrics['pass@8'])} | "
                    f"pass@16={format_metric(metrics['pass@16'])} | "
                    f"pass@32={format_metric(metrics['pass@32'])} | "
                    f"pass@64={format_metric(metrics['pass@64'])}"
                )

            metrics = dataset_result["pass@k"]
            print(
                f"  [{dataset_name}] "
                f"pass@1={format_metric(metrics['pass@1'])} | "
                f"pass@4={format_metric(metrics['pass@4'])} | "
                f"pass@8={format_metric(metrics['pass@8'])} | "
                f"pass@16={format_metric(metrics['pass@16'])} | "
                f"pass@32={format_metric(metrics['pass@32'])} | "
                f"pass@64={format_metric(metrics['pass@64'])}"
            )
            wait_for_ports_free(target_gpu_ids, timeout=60)

    finally:
        print("\nShutting down workers...")
        shutdown_workers(processes, work_queues)

    save_results(output_path, args.model_path, datasets_config, all_results, args)
    save_summary(output_path, args.model_path, all_results, args)

    print(f"\n{'=' * 72}")
    print(f"{'Dataset':<12} {'Runs':>8} {'pass@1':>10} {'pass@4':>10} {'pass@8':>10} {'pass@16':>10} {'pass@32':>10} {'pass@64':>10}")
    print(f"{'-' * 72}")
    for dataset_cfg in datasets_config:
        dataset_result = all_results[dataset_cfg['dataset_name']]
        metrics = dataset_result["pass@k"]
        print(
            f"{dataset_cfg['dataset_name']:<12} "
            f"{dataset_result['completed_num_runs']:>8} "
            f"{format_metric(metrics['pass@1']):>10} "
            f"{format_metric(metrics['pass@4']):>10} "
            f"{format_metric(metrics['pass@8']):>10} "
            f"{format_metric(metrics['pass@16']):>10} "
            f"{format_metric(metrics['pass@32']):>10} "
            f"{format_metric(metrics['pass@64']):>10}"
        )
    macro = compute_macro_pass_at_k(all_results)
    print(f"{'-' * 72}")
    print(
        f"{'macro_avg':<12} {'-':>8} "
        f"{format_metric(macro['pass@1']):>10} "
        f"{format_metric(macro['pass@4']):>10} "
        f"{format_metric(macro['pass@8']):>10} "
        f"{format_metric(macro['pass@16']):>10} "
        f"{format_metric(macro['pass@32']):>10} "
        f"{format_metric(macro['pass@64']):>10}"
    )
    print(f"{'=' * 72}")
    print(f"Saved to: {output_path}")
    print(f"Summary:  {output_path.replace('.json', '_summary.json')}")


if __name__ == "__main__":
    main()