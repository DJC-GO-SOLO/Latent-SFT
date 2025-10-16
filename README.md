# Latent Reasoning in LLMs as a Vocabulary-Space Superposition
This is the official code repository for the latent-SFT paper. We provide the source code, which is very easy to hack into, for your convenience.
![](https://github.com/DJC-GO-SOLO/Latent-SFT/blob/main/figs/comparison.png)

## Abstract
Large language models (LLMs) demonstrate strong reasoning abilities with chain-of-thought prompting, but explicit reasoning introduces substantial computational overhead. Recent work on latent reasoning reduces this cost by reasoning in latent space without explicit supervision, but performance drops significantly. Our preliminary experiments suggest that this degradation stems from the unstructured latent space, which makes fitting latent tokens difficult. To address this, we restrict the latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Once latent reasoning concludes, it collapses into an eigenstate of explicit reasoning to yield the final answer. Based on this idea, we propose Latent-SFT, a two-stage learning framework. In the first stage, we design two specialized attention masks to guide the Latent Token Encoder in generating latent tokens, allowing the LLM to produce the correct answer conditioned on them. In the second stage, the Latent Token Encoder is discarded, and the LLM is directly trained to generate these latent tokens autonomously for latent reasoning, optimized with KL and CE losses. Latent-SFT sets a new state of the art on GSM8k, matching explicit SFT performance while cutting reasoning chains by up to 4× and outperforming prior latent methods. On Math500 and AIME24, lexical probability–based latent reasoning also clearly surpasses hidden-state–based approaches. Our metrics of effective compression rate and effective global parallelism further show that latent reasoning is both the compression of a single path and the superposition of multiple paths.

## Method
![](https://github.com/DJC-GO-SOLO/Latent-SFT/blob/main/figs/overview.png)

## Usage
### Data Preparation
First, download the required datasets from the official [Hugging Face library](https://huggingface.co/datasets/DJCheng/Latent-SFT-Data/tree/main) and place them in the `./data` directory. This includes both the training and evaluation data needed for the model.

### Dependent Libraries
```
pip install -r requirements.txt
```
> **Note:** Installing the flash-attn library is required and may take some time, depending on your system configuration.

### COT-SFT Model Weights
Currently, training is supported for two models: **llama3.2-instruct-1B** and **Deepseek-distill-qwen-7B**.
The initialization weights can be downloaded from the following links:
- [*llama3.2-instruct-1B-COT-SFT*](https://huggingface.co/DJCheng/Latent-SFT-Llama3.2-Instruct-1B-COT-SFT/tree/main)
- [*Deepseek-distill-qwen-7B*](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

Since the **Deepseek-distill-qwen-7B** model has already been trained on the inference dataset, its original weights can be used directly without additional fine-tuning.

### Training phases
The training pipeline consists of two sequential phases. In Phase 1, latent tokens are generated. In Phase 2, the LLM is trained to autonomously generate these latent tokens, enabling latent planning during inference.
#### Phase 1: Generating Latent Tokens
Due to the limitations of *soft embeddings*, the learning process becomes more challenging. To address this, we adopt a three-step optimization strategy to facilitate model convergence.
##### Step 1: Training the encoder
The training script to be executed for this part is:
```
cd script
./run_distill_stage1_soft_embedding_encoder.sh
```
> Pay close attention to the path-related parameters in the script, including the path for saving the model and the path for loading the initial model weights.
> Additionally, you may adjust various training hyperparameters (e.g., learning rate, batch size) to improve model convergence.

##### Step 2: Training the decoder
The training script to be executed for this part is:
```
cd script
./run_distill_stage1_soft_embedding_decoder.sh
```
> This Step requires the best model weights from **Steps 1**.
##### Step 3: Joint Optimization
The training script to be executed for this part is:
```
cd script
./run_distill_stage1_soft_embedding_union.sh
```
> **Note:** This section saves only the LoRA adapter weights. To obtain the full model weights, use the `merge_lora.py` script to merge the adapter with the base model.
> This Step requires the best model weights from **Steps 1** and **Step 2**.

#### Phase 2: Learning Latent Tokens
In this phase, we first use the encoder trained in Phase 1 to generate latent tokens—represented as probability distributions over the vocabulary—for each training sample. Then, the trained decoder is used to learn both the latent tokens and the final output answer.

##### Step 1: Caching latent tokens
If the model uses **LoRA weights** (i.e., the result saved in **Step 3 of Phase 1**), run the `generate_latent_soft_label_union_hf.py` script directly.

⚠️ Make sure the compression rate parameter is consistent with the one used during training.

If the model uses **full weights** (i.e., generated in **Step 1 of Phase 1**), run the `generate_latent_soft_label_hf.py` script instead.

##### Step 2: Training the decoder
The training script to be executed for this part is:
```
cd script
./run_distill_stage2_soft_embedding.sh
```

### Model Evaluation
We provide evaluations on six mathematical reasoning datasets — **GSM8k**, **GSM-Hard**, **SVAMP**, **MultiArith**, **Math500**, and **AIME24** — using two implementation frameworks: **Transformers** and **SGLang**.

#### Transformers
Evaluation using the **Transformers** framework is relatively slow and is best suited for **small models** and **short datasets**.

The following evaluation scripts are provided:

📘 **Phase 1**:
- `eval_soft_embedding_encoder_hf.py` – for evaluating **encoder** training
- `eval_soft_embedding_decoder_hf.py` – for evaluating **decoder** training
- `eval_soft_embedding_union_hf.py` – for evaluating the **joint encoder-decoder** model

📗 **Phase 2**:
- `eval_soft_embedding_latent_model_hf.py` – for evaluating **in-distribution** datasets
- `eval_ood_soft_embedding_latent_model.py` – for evaluating **out-of-distribution** datasets

#### SGLang
The **SGLang** framework is used to enable **faster evaluation**, especially for **larger models** and **longer sequences**. However, this speed-up may come at the cost of slightly reduced accuracy in certain cases. Some parts of the code have been modified based on the original [Soft-Thinking](https://github.com/eric-ai-lab/Soft-Thinking) library to support our specific evaluation workflow.

First, navigate to the modified SGLang directory. Then, create a new virtual environment and install the required dependencies. ⚠️ **It is strongly recommended to keep this environment *separate* from the training environment to avoid dependency conflicts.**
```
conda create -n sg python=3.11 -y && conda activate sg
pip install --upgrade pip
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver
pip install flash_attn --no-build-isolation

cd sglang_latent_reasoning_pkg
pip install -e "python[all]"
cd ..
```
Next, run the evaluation file directly, which will output the inference results of the model. Note that you need to modify the corresponding path.
```
python eval_soft_embedding_sglang.py
```
Finally, the score is obtained through the scoring file. Note that you need to modify the corresponding path.
```
python get_math_score.py
```

### Best Model Weights
The trained model weights are available at the following links:
- [Llama3.2-Instruct-1B-Latent(2)](https://huggingface.co/DJCheng/Llama3.2-Instruct-1B-Latent-2/tree/main)
- [Llama3.2-Instruct-1B-Latent(4)](https://huggingface.co/DJCheng/Llama3.2-Instruct-1B-Latent-4/tree/main)

⚠️ Due to certain constraints, the Deepseek-distill-qwen-7B-Latent model weights are not yet released, but will be made available soon.
