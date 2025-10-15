from transformers.trainer import Trainer
from transformers import AutoModel, AutoModelForCausalLM
from peft import AutoPeftModel, PeftModel
import torch
from typing import *
import os
import torch.distributed as dist
from transformers.trainer import *
from peft import AutoPeftModel, PeftModel

class Stage1EncoderTrainer(Trainer):

    def __init__(self, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info("Saving model checkpoint to %s", output_dir)
        print("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            if self.model.lora_tune:
                
                if self.is_world_process_zero():
                    self.model.save(os.path.join(output_dir, 'lora_adapter'))

                    model = AutoModel.from_pretrained(os.path.join(self.model.save_path, 'base_model'),
                        torch_dtype=torch.bfloat16,
                        use_cache=False,
                        trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(
                        model, 
                        os.path.join(output_dir, 'lora_adapter')
                    )
                    model = model.merge_and_unload()
                    # shutil.rmtree(os.path.join(output_dir, 'base_model')) # remove lora_adapter
                    model.save_pretrained(os.path.join(output_dir, 'hf'))
                    self.model.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))
                    del model
            else:
                self.model.save_pretrained(os.path.join(output_dir, 'hf'))

        # if self.tokenizer is not None and self.is_world_process_zero():
        #     self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

class Stage1DecoderTrainer(Trainer):

    def __init__(self, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info("Saving model checkpoint to %s", output_dir)
        print("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            if self.model.lora_tune:
                
                if self.is_world_process_zero():
                    self.model.save(os.path.join(output_dir, 'lora_adapter'))

                    model = AutoModelForCausalLM.from_pretrained(os.path.join(self.model.save_path, 'base_model'),
                        torch_dtype=torch.bfloat16,
                        use_cache=False,
                        trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(
                        model, 
                        os.path.join(output_dir, 'lora_adapter')
                    )
                    model = model.merge_and_unload()
                    # shutil.rmtree(os.path.join(output_dir, 'base_model')) # remove lora_adapter
                    model.save_pretrained(os.path.join(output_dir, 'hf'))
                    self.model.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))
                    del model
            else:
                self.model.save_pretrained(os.path.join(output_dir, 'hf'))

        # if self.tokenizer is not None and self.is_world_process_zero():
        #     self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

class Stage1UnionTrainer(Trainer):

    def __init__(self, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info("Saving model checkpoint to %s", output_dir)
        print("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            if self.model.lora_tune:
                
                if self.is_world_process_zero():
                    self.model.save(os.path.join(output_dir, 'lora_adapter'))

            else:
                self.model.save_pretrained(os.path.join(output_dir, 'hf'))

        # if self.tokenizer is not None and self.is_world_process_zero():
        #     self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
