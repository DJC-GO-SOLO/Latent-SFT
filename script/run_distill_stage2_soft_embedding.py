import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import logging
from pathlib import Path

from transformers import (
    AutoConfig, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from src.modeling.modeling_latent import (
    LatentSFTStage2SoftEmbedding
)

from src.stage2_soft_embedding.arguments import (
    ModelArguments, 
    DataArguments, 
    Stage2TrainingArguments as TrainingArguments
)

from src.stage2_soft_embedding.data import (
    Stage2Dataset, 
    DataCollatorForDynamicPadding,
)

from src.stage2_soft_embedding.trainer import (
    Stage2Trainer
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

    model = LatentSFTStage2SoftEmbedding(
        latent_model_path=model_args.latent_model_path,
        ce_w=model_args.ce_w,
        kl_w=model_args.kl_w,
        bfloat16=model_args.bfloat16,
        use_flash_attention_2=model_args.use_flash_attention_2,
        lora_tune=training_args.lora_tune,
        lora_path=training_args.lora_path,
        lora_rank=training_args.lora_rank,
        lora_dropout=training_args.lora_dropout,
        save_path=training_args.output_dir,
        training = training_args.training,
    )

    train_dataset = Stage2Dataset(path=data_args.train_data_path,
        train_latent_soft_label_path=data_args.train_latent_soft_label_path,
        args=data_args, 
        model=model
    )

    

    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        data_collator=DataCollatorForDynamicPadding(model.tokenizer.pad_token_id),
        tokenizer=model.tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()