import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import logging
from pathlib import Path
from torch.optim import AdamW
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from src.modeling.modeling_compress import (
    LatentSFTStage1Union
)

from src.compress_dymatic_soft_embedding.arguments import (
    ModelArguments, 
    DataArguments, 
    Stage1TrainingArguments as TrainingArguments
)

from src.compress_dymatic_soft_embedding.data import (
    Stage1Dataset, 
    DataCollatorForDynamicPadding,
)

from src.compress_dymatic_soft_embedding.trainer import (
    Stage1UnionTrainer
)

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
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    print('================================')
    print(training_args.seed)
    print('================================')
    set_seed(training_args.seed)

    model = DistillLatentStage1DynamicSoftEmbedding3(
        encoder_name_or_path=model_args.encoder_name_or_path,
        decoder_name_or_path=model_args.decoder_name_or_path,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
    )

    train_dataset = Stage1Dataset(path=data_args.train_data_path,
        args=data_args, 
        model=model
    )
    

    trainer = Stage1UnionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForDynamicPadding(model.tokenizer.pad_token_id,model.compress_token_id, model.latent_token_ids[-1]),
        tokenizer=model.tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    train_result = trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        model.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()