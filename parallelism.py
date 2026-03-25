"""
Core idea:
1. Do not force position-wise alignment between latent step t and explicit segment t.
2. For each candidate latent step, match it to the closest unused reference segment latent.
3. Compute path scores from the decoder probability mass assigned to the matched segments.
4. Compute N_eff from these matched path scores.
5. Compute N_eff_naive by repeating the same procedure with random no-repeat matching.
6. Report the final score as N_eff / N_eff_naive.

Important assumption:
- `reference_segment_latents_base` must be precomputed from the same segment partition
  used by `segment_token_ids_by_steps(...)` below. That is, the reference latent z_{m,s}
  and the explicit token segment S_{m,s} must correspond to the same segment.
- The path order in `reference_segment_latents_base` must match the path order returned by
  `collect_explicit_paths(...)`.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer


@dataclass
class Config:
    data_jsonl_path: str = ""
    tokenizer_path: str = ""
    candidate_latents_base: str = ""
    candidate_logits_base: str = ""
    reference_segment_latents_base: str = ""
    output_json_path: str = ""

    topk: Optional[int] = 100
    distance_metric: str = "cosine"   # "cosine" or "l2"
    temperature: float = 1.0
    baseline_trials: int = 64
    random_seed: int = 0


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def collect_explicit_paths(example: Dict[str, Any]) -> List[str]:
    """
    Minimal path collection:
    - primary path: example["solution"]
    - alternative paths: example["gen_solutions"]
    Duplicate paths are removed exactly after simple normalization.
    """
    paths: List[str] = []

    if isinstance(example.get("solution"), str) and example["solution"].strip():
        paths.append(example["solution"])

    gen_solutions = example.get("gen_solutions", [])
    if isinstance(gen_solutions, list):
        for text in gen_solutions:
            if isinstance(text, str) and text.strip():
                paths.append(text)

    unique_paths: List[str] = []
    seen = set()
    for path in paths:
        key = normalize_text(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def find_tensor_path(base_path: Union[str, Path], idx: int) -> Optional[Path]:
    """
    Supported naming:
    - <base>.pt       (tries this only for idx == 0)
    - <base>_0.pt
    - <base>0.pt
    """
    p = Path(base_path)
    base_no_suffix = p.with_suffix("")
    suffix = p.suffix or ".pt"

    candidates = [
        Path(str(base_no_suffix) + f"_{idx}{suffix}"),
        Path(str(base_no_suffix) + f"{idx}{suffix}"),
    ]
    if idx == 0:
        candidates.insert(0, p)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_tensor_by_index(base_path: Union[str, Path], idx: int) -> Optional[torch.Tensor]:
    tensor_path = find_tensor_path(base_path, idx)
    if tensor_path is None:
        return None

    obj = torch.load(str(tensor_path), map_location="cpu")
    if isinstance(obj, list):
        obj = torch.stack([torch.as_tensor(x) for x in obj], dim=0)
    return torch.as_tensor(obj)


def detect_is_prob(row_1d: torch.Tensor) -> bool:
    x = row_1d.float()
    if (x >= 0).all() and torch.isfinite(x).all():
        s = float(x.sum().item())
        return abs(s - 1.0) < 2e-2
    return False


def ensure_probs(row_1d: torch.Tensor, looks_like_probs: bool) -> torch.Tensor:
    x = row_1d.float()
    if looks_like_probs:
        s = x.sum()
        if s > 0 and torch.isfinite(s):
            return x / s
    return torch.softmax(x, dim=-1)


def adaptive_segment_indices(length: int, num_steps: int) -> List[Tuple[int, int]]:
    return [((t * length) // num_steps, ((t + 1) * length) // num_steps) for t in range(num_steps)]


def segment_token_ids_by_steps(token_ids: Sequence[int], num_steps: int) -> List[List[int]]:
    spans = adaptive_segment_indices(len(token_ids), num_steps)
    return [list(token_ids[st:ed]) for st, ed in spans]


def latent_distance_matrix(
    candidate_latents: torch.Tensor,   # [T, d]
    reference_latents: torch.Tensor,   # [T', d]
    metric: str = "cosine",
    eps: float = 1e-12,
) -> torch.Tensor:
    cand = torch.as_tensor(candidate_latents, dtype=torch.float32)
    ref = torch.as_tensor(reference_latents, dtype=torch.float32)

    if cand.ndim != 2 or ref.ndim != 2:
        raise ValueError("candidate_latents and reference_latents must be 2D.")
    if cand.shape[1] != ref.shape[1]:
        raise ValueError("Latent dimensions do not match.")

    if metric.lower() == "cosine":
        cand = cand / cand.norm(dim=-1, keepdim=True).clamp_min(eps)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(eps)
        return 1.0 - cand @ ref.transpose(0, 1)

    if metric.lower() in {"l2", "euclidean"}:
        diff = cand[:, None, :] - ref[None, :, :]
        return torch.norm(diff, p=2, dim=-1)

    raise ValueError("distance_metric must be 'cosine' or 'l2'.")


def greedy_unique_assignment(dist_mat: torch.Tensor) -> List[int]:
    """
    Implement the matching rule:
    for each candidate latent step, choose the closest unused reference segment.
    """
    dist = torch.as_tensor(dist_mat, dtype=torch.float32)
    num_rows, num_cols = dist.shape

    assignment = [-1] * num_rows
    used_cols = set()

    for t in range(num_rows):
        best_s = -1
        best_dist = None
        for s in range(num_cols):
            if s in used_cols:
                continue
            d = float(dist[t, s].item())
            if best_dist is None or d < best_dist:
                best_dist = d
                best_s = s
        if best_s >= 0:
            assignment[t] = best_s
            used_cols.add(best_s)

    return assignment


def random_unique_assignment(
    num_rows: int,
    num_cols: int,
    generator: torch.Generator,
) -> List[int]:
    """
    Naive baseline:
    random no-repeat matching instead of best matching.
    """
    assignment = [-1] * num_rows
    if num_cols <= 0:
        return assignment

    perm = torch.randperm(num_cols, generator=generator).tolist()
    usable = min(num_rows, num_cols)
    for t in range(usable):
        assignment[t] = perm[t]
    return assignment


def prepare_prob_rows_and_topk_masks(
    logits_or_probs: torch.Tensor,
    topk: Optional[int],
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    looks_like_probs = detect_is_prob(logits_or_probs[0])
    num_steps, vocab_size = logits_or_probs.shape

    prob_rows: List[torch.Tensor] = []
    topk_masks: Optional[List[torch.Tensor]] = [] if topk is not None else None
    k_eff = min(int(topk), vocab_size) if topk is not None else None

    for t in range(num_steps):
        row = ensure_probs(logits_or_probs[t], looks_like_probs)
        prob_rows.append(row)
        if topk_masks is not None and k_eff is not None:
            _, idx = torch.topk(row, k_eff, largest=True, sorted=False)
            mask = torch.zeros_like(row, dtype=torch.bool)
            mask[idx] = True
            topk_masks.append(mask)

    return prob_rows, topk_masks


def assignment_log_mass_score(
    prob_rows: List[torch.Tensor],
    assignment: List[int],
    segment_token_ids: List[List[int]],
    *,
    topk_masks: Optional[List[torch.Tensor]] = None,
    eps: float = 1e-12,
) -> float:
    """
    Score one reference path after a matching is fixed.

    We keep the same intuition as the original metric:
    a path is better supported if the candidate latent decoder places higher probability
    mass on the tokens of the matched segments.
    """
    total_score = 0.0
    num_steps = len(assignment)
    vocab_size = prob_rows[0].shape[0]

    for t, matched_segment in enumerate(assignment):
        if matched_segment < 0 or matched_segment >= len(segment_token_ids):
            total_score += math.log(eps)
            continue

        ids = segment_token_ids[matched_segment]
        if len(ids) == 0:
            total_score += math.log(eps)
            continue

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        valid_ids = ids_tensor[(ids_tensor >= 0) & (ids_tensor < vocab_size)]
        if valid_ids.numel() == 0:
            total_score += math.log(eps)
            continue

        row = prob_rows[t]
        if topk_masks is None:
            mass = float(row.index_select(0, valid_ids).sum().item())
        else:
            keep = topk_masks[t][valid_ids]
            mass = float(row.index_select(0, valid_ids)[keep].sum().item()) if keep.any() else 0.0

        total_score += math.log(mass + eps)

    return total_score / float(max(1, num_steps))


def posterior_from_scores(scores: Sequence[float], temperature: float, eps: float = 1e-12) -> torch.Tensor:
    score_tensor = torch.tensor(list(scores), dtype=torch.float64)
    score_tensor = score_tensor / float(max(1e-8, temperature))
    max_logit = float(torch.max(score_tensor))
    exps = torch.exp(score_tensor - max_logit)
    return exps / torch.clamp(exps.sum(), min=eps)


def neff_from_scores(scores: Sequence[float], temperature: float, eps: float = 1e-12) -> float:
    posterior = posterior_from_scores(scores, temperature=temperature, eps=eps)
    entropy = -float(torch.sum(posterior * torch.log(posterior + eps)))
    return math.exp(entropy)


@torch.no_grad()
def compute_sample_parallelism_score(
    candidate_latents: torch.Tensor,               # [T, d]
    logits_or_probs: torch.Tensor,                 # [T, V]
    explicit_paths_ids: List[List[int]],           # M paths
    reference_segment_latents: List[torch.Tensor], # M tensors, each [T', d]
    *,
    topk: Optional[int],
    distance_metric: str,
    temperature: float,
    baseline_trials: int,
    random_seed: int,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute:
    - N_eff from best latent-to-latent matching
    - N_eff_naive from random no-repeat matching
    - final_score = N_eff / N_eff_naive
    """
    if len(explicit_paths_ids) != len(reference_segment_latents):
        raise ValueError("explicit_paths_ids and reference_segment_latents must have the same number of paths.")

    candidate_latents = torch.as_tensor(candidate_latents, dtype=torch.float32)
    logits_or_probs = torch.as_tensor(logits_or_probs)
    num_steps = min(int(candidate_latents.shape[0]), int(logits_or_probs.shape[0]))

    candidate_latents = candidate_latents[:num_steps]
    logits_or_probs = logits_or_probs[:num_steps]
    prob_rows_all, topk_masks_all = prepare_prob_rows_and_topk_masks(logits_or_probs, topk=topk)

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))

    matched_path_scores: List[float] = []
    naive_trial_path_scores: List[List[float]] = [[] for _ in range(max(1, baseline_trials))]

    for path_ids, ref_latents in zip(explicit_paths_ids, reference_segment_latents):
        ref_latents = torch.as_tensor(ref_latents, dtype=torch.float32)
        aligned_steps = min(num_steps, int(ref_latents.shape[0]))
        if aligned_steps <= 0:
            raise ValueError("Each reference path must contain at least one latent segment.")

        cand_lat = candidate_latents[:aligned_steps]
        ref_lat = ref_latents[:aligned_steps]
        prob_rows = prob_rows_all[:aligned_steps]
        topk_masks = None if topk_masks_all is None else topk_masks_all[:aligned_steps]
        path_segments = segment_token_ids_by_steps(path_ids, aligned_steps)

        dist_mat = latent_distance_matrix(cand_lat, ref_lat, metric=distance_metric, eps=eps)
        best_assignment = greedy_unique_assignment(dist_mat)

        matched_score = assignment_log_mass_score(
            prob_rows,
            best_assignment,
            path_segments,
            topk_masks=topk_masks,
            eps=eps,
        )
        matched_path_scores.append(matched_score)

        for trial_idx in range(max(1, baseline_trials)):
            random_assignment = random_unique_assignment(aligned_steps, aligned_steps, generator)
            naive_score = assignment_log_mass_score(
                prob_rows,
                random_assignment,
                path_segments,
                topk_masks=topk_masks,
                eps=eps,
            )
            naive_trial_path_scores[trial_idx].append(naive_score)

    n_eff = neff_from_scores(matched_path_scores, temperature=temperature, eps=eps)

    naive_neff_values = [
        neff_from_scores(path_scores, temperature=temperature, eps=eps)
        for path_scores in naive_trial_path_scores
    ]
    n_eff_naive = sum(naive_neff_values) / float(len(naive_neff_values))
    final_score = n_eff / max(eps, n_eff_naive)

    return {
        "N_eff": float(n_eff),
        "N_eff_naive": float(n_eff_naive),
        "final_score": float(final_score),
    }


def evaluate_dataset(config: Config) -> Dict[str, Any]:
    if not config.data_jsonl_path:
        raise ValueError("Please fill config.data_jsonl_path.")
    if not config.tokenizer_path:
        raise ValueError("Please fill config.tokenizer_path.")
    if not config.candidate_latents_base:
        raise ValueError("Please fill config.candidate_latents_base.")
    if not config.candidate_logits_base:
        raise ValueError("Please fill config.candidate_logits_base.")
    if not config.reference_segment_latents_base:
        raise ValueError("Please fill config.reference_segment_latents_base.")
    if not config.output_json_path:
        raise ValueError("Please fill config.output_json_path.")

    data = read_jsonl(config.data_jsonl_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    sample_results: List[Dict[str, Any]] = []
    n_eff_values: List[float] = []
    n_eff_naive_values: List[float] = []
    final_scores: List[float] = []

    for sample_index, example in enumerate(data):
        explicit_paths_text = collect_explicit_paths(example)
        if len(explicit_paths_text) < 2:
            continue

        explicit_paths_ids = [
            tokenizer(
                text,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )["input_ids"]
            for text in explicit_paths_text
        ]

        candidate_latents = load_tensor_by_index(config.candidate_latents_base, sample_index)
        candidate_logits = load_tensor_by_index(config.candidate_logits_base, sample_index)
        if candidate_latents is None or candidate_logits is None:
            continue

        precomputed = load_tensor_by_index(config.reference_segment_latents_base, sample_index)
        if precomputed is None:
            continue

        if isinstance(precomputed, torch.Tensor) and precomputed.ndim == 3:
            reference_segment_latents = [precomputed[m] for m in range(precomputed.shape[0])]
        else:
            raise ValueError(
                "Expected precomputed reference segment latents to be a tensor with shape [M, T, d]."
            )

        result = compute_sample_parallelism_score(
            candidate_latents=candidate_latents,
            logits_or_probs=candidate_logits,
            explicit_paths_ids=explicit_paths_ids,
            reference_segment_latents=reference_segment_latents,
            topk=config.topk,
            distance_metric=config.distance_metric,
            temperature=config.temperature,
            baseline_trials=config.baseline_trials,
            random_seed=config.random_seed + sample_index,
        )

        sample_results.append({
            "sample_index": sample_index,
            "num_paths": len(explicit_paths_text),
            **result,
        })
        n_eff_values.append(result["N_eff"])
        n_eff_naive_values.append(result["N_eff_naive"])
        final_scores.append(result["final_score"])

    summary = {
        "num_processed_samples": len(sample_results),
        "mean_N_eff": float(sum(n_eff_values) / max(1, len(n_eff_values))),
        "mean_N_eff_naive": float(sum(n_eff_naive_values) / max(1, len(n_eff_naive_values))),
        "mean_final_score": float(sum(final_scores) / max(1, len(final_scores))),
        "samples": sample_results,
    }

    output_path = Path(config.output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    cfg = Config(
        data_jsonl_path="",
        tokenizer_path="",
        candidate_latents_base="",
        candidate_logits_base="",
        output_json_path="",
        reference_segment_latents_base="",
        topk=100,
        distance_metric="cosine",
        temperature=1.0,
        baseline_trials=64,
        random_seed=0,
    )

    results = evaluate_dataset(cfg)
    print(
        f"Processed {results['num_processed_samples']} samples | "
        f"mean_N_eff = {results['mean_N_eff']:.4f} | "
        f"mean_N_eff_naive = {results['mean_N_eff_naive']:.4f} | "
        f"mean_final_score = {results['mean_final_score']:.4f}"
    )
