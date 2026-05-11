import logging
from torch._tensor import Tensor
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda
import torch.nn.functional as F

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of latent reasoning
        # ==========
        enable_latent: bool = False,
        # ==========
        # end of latent reasoning
        # ==========
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits
        # ==========
        # begin of latent reaonsing
        # ==========
        if enable_latent:
            add_noise_gumbel_softmax = sampling_info.add_noise_gumbel_softmax.any().item()
            
            if add_noise_gumbel_softmax:
                logits_f32 = logits.float()
                full_log_probs = logits_f32.log_softmax(dim=-1)
                
                probs = logits_f32.softmax(dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                top_p_val = sampling_info.top_ps.view(-1, 1)
                sorted_mask = (cumulative_probs - sorted_probs) < top_p_val
                # [Constraint] Ensure at least Top-K tokens are valid for sampling
                # Corresponds to "When Top-P < Top-K, take Top-K"
                sorted_mask[:, :sampling_info.max_topk] = True
                mask = torch.zeros_like(logits_f32, dtype=torch.bool).scatter_(1, sorted_indices, sorted_mask)
                
                sampling_log_probs = full_log_probs.masked_fill(~mask, float('-inf'))
                # Sample standard Gumbel noise first. One-sided noise is kept as
                # an opt-in transform for experiments that require it.
                gumbels = -torch.empty_like(sampling_log_probs).exponential_().log()
                gumbels = gumbels.clamp(-1.5, 3)
                use_one_sided_gumbel_noise = sampling_info.use_one_sided_gumbel_noise.view(-1, 1)
                if use_one_sided_gumbel_noise.any().item():
                    one_sided_gumbels = gumbels - (-1.5)
                    gumbels = torch.where(
                        use_one_sided_gumbel_noise,
                        one_sided_gumbels,
                        gumbels,
                    )

                gumbels = sampling_info.noise_scales[0] * gumbels

                
                topk_gumbels_scores = sampling_log_probs + gumbels
                
                # Extract Top-K indices based on sampling scores
                # topk_gumbels here refers to the scores (logp + noise)
                # Note: logits_output.topk_gumbels usually expects the value that was used for selection
                topk_gumbels, topk_indices = torch.topk(topk_gumbels_scores, k=sampling_info.max_topk, dim=-1)
                
                topk_logps = full_log_probs.gather(-1, topk_indices)
                
                # Keep original logits if needed
                topk_logits_original = logits_f32.gather(-1, topk_indices)
                # Recover the exact Gumbel noise for the selected Top-K
                # Direct gather from the source noise tensor (more precise)
                gumbel_noise_topk = gumbels.gather(-1, topk_indices)



                topk_probs = (topk_gumbels / sampling_info.gumbel_softmax_temperatures[0]).softmax(-1)   

                log_prob_noise_topk = (-gumbel_noise_topk - (-gumbel_noise_topk).exp()).sum(-1) / sampling_info.max_topk

                logits_output.topk_gumbels = topk_gumbels # gumbels
                # logits_output.topk_original_probs = torch.gather(logits_output.topk_original_probs, dim=1, index=sorted_idx)
                logits_output.topk_probs = topk_probs #gumbel probs
                logits_output.topk_indices = topk_indices


                topk_logits, topk_indices = torch.topk(logits_f32, k=sampling_info.max_topk, dim=-1)


                mask = (sampling_info.latent_modes == True) & (topk_indices[:,0] != 524) & sampling_info.add_noise_gumbel_softmax
                logits_output.topk_probs[:] = torch.where(
                    mask.unsqueeze(-1),
                    logits_output.topk_probs,
                    torch.softmax(topk_logits, dim=-1),
                )
                logits_output.topk_indices[:] = torch.where(
                    mask.unsqueeze(-1),
                    logits_output.topk_indices,
                    topk_indices,
                )
                latent_probs = torch.softmax(logits/sampling_info.temperatures[0], dim=-1)
                topk_original_probs, _ = torch.topk(latent_probs, k=sampling_info.max_topk, dim=-1)
                logits_output.topk_original_probs = topk_original_probs
                logits_output.topk_original_indices = topk_indices
            else:
                topk_logits, topk_indices = torch.topk(logits, k=sampling_info.max_topk, dim=-1)
                logits_output.topk_gumbels  = topk_logits.clone()
                logits_output.topk_probs = torch.softmax(topk_logits, dim=-1)
                logits_output.topk_original_probs,_ =  torch.topk(torch.softmax(logits/sampling_info.temperatures[0], dim=-1), k=sampling_info.max_topk, dim=-1)
                logits_output.topk_indices = topk_indices
                logits_output.topk_original_indices = topk_indices.clone()


            latent_batch_next_token_ids = logits_output.topk_indices[:,0]
            
        # ==========
        # end of latent reaonsing
        # ==========

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)

                max_top_k_round, batch_size = 32, probs.shape[0]
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    # Check Nan will throw exception, only check when crash_on_warnings is True
                    check_nan = self.use_nan_detection and crash_on_warnings()
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                        check_nan=check_nan,
                    )

            elif global_server_args_dict["sampling_backend"] == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        batch_next_token_ids[:] = torch.where(
            sampling_info.latent_modes,
            latent_batch_next_token_ids,
            batch_next_token_ids,
        )
        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            # if any(x > 0 for x in top_logprobs_nums):
            #     (
            #         logits_output.next_token_top_logprobs_val,
            #         logits_output.next_token_top_logprobs_idx,
            #     ) = get_top_logprobs(logprobs, top_logprobs_nums)

            # if any(x is not None for x in token_ids_logprobs):
            #     (
            #         logits_output.next_token_token_ids_logprobs_val,
            #         logits_output.next_token_token_ids_logprobs_idx,
            #     ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            # logits_output.next_token_logprobs = logprobs[
            #     torch.arange(len(batch_next_token_ids), device=sampling_info.device),
            #     batch_next_token_ids,
            # ]
            if enable_latent and sampling_info.add_noise_gumbel_softmax[0]:
                logits_output.next_token_gumbel_logprobs = log_prob_noise_topk
                logits_output.next_token_logprobs = logprobs[
                    torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                    batch_next_token_ids,
                ]
            else:
                logits_output.next_token_gumbel_logprobs = logits_output.next_token_logprobs = logprobs[
                    torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                    batch_next_token_ids,
                ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids


       
    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx