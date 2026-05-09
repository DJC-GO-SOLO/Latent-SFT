"""Stage-1 Decoder distillation entry point.

This script trains the Stage-1 *decoder* of the latent-SFT top-k soft-embedding
pipeline. It expects a previously trained Stage-1 *encoder* checkpoint (produced
by ``run_distill_stage1_encoder.py``) whose path is supplied via
``--encoder_name_or_path``. The base decoder backbone is loaded from
``--decoder_name_or_path``.

The module is intended to be launched through
``script/run_distill_stage1_decoder.sh`` with ``torchrun``; all
CLI arguments are parsed via :class:`transformers.HfArgumentParser` from the
dataclasses defined in ``src.stage1.arguments``.
"""

import logging
import os
import sys
from pathlib import Path

# Make ``src`` importable when the script is executed directly.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from transformers import HfArgumentParser, set_seed  # noqa: E402

from src.stage1.arguments import (  # noqa: E402
    DataArguments,
    ModelArguments,
    Stage1TrainingArguments as TrainingArguments,
)
from src.stage1.data import (  # noqa: E402
    DataCollatorForDynamicPadding,
    Stage1Dataset,
)
from src.stage1.trainer import Stage1DecoderTrainer  # noqa: E402
from src.modeling.modeling_stage1 import LatentSFTStage1Decoder  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not"
            + " empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging (info on rank-0 only, warnings elsewhere).
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,"
        + " 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Seed every library that supports it.
    logger.info("Setting global random seed to %d", training_args.seed)
    set_seed(training_args.seed)

    model = LatentSFTStage1Decoder(
        encoder_name_or_path=model_args.encoder_name_or_path,
        decoder_name_or_path=model_args.decoder_name_or_path,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        topk_interpolation=model_args.topk_interpolation,
    )

    train_dataset = Stage1Dataset(
        path=data_args.train_data_path,
        args=data_args,
        model=model,
    )

    trainer = Stage1DecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForDynamicPadding(
            model.tokenizer.pad_token_id,
            model.compress_token_id,
            model.latent_token_ids[-1],
        ),
        tokenizer=model.tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()

    if trainer.is_world_process_zero():
        model.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()