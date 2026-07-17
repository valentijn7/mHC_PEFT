# Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning

This repository contains the code for the paper **“Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning”**.

We study manifold-constrained hyper-connections (mHC) as parameter-efficient finetuning modules for frozen Transformer language models. Instead of only adapting weights or activations, mHC modifies the residual-stream structure around frozen OLMo-2 layers. This codebase supports several mHC variants, standard PEFT baselines, training on Tulu instruction data, held-out perplexity evaluation, and downstream benchmarking with `lm-evaluation-harness`.

Paper URL and BibTeX will be added after release.

## Installation

The repository is built around the conda environment in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate FoMo
````

Optional Hugging Face authentication:

```bash
huggingface-cli login
```

The main experiments use OLMo-2 models from Hugging Face and train with bfloat16 on GPU.

## Quick start

Run a small smoke test with dynamic KromHC:

```bash
python main.py \
  run_name=smoke_kromhc \
  method.selected_method=kromhc \
  train.max_train_steps=10 \
  data.max_train_samples=128 \
  data.validation_samples=64 \
  data.test_samples=64
```

Train dynamic KromHC on OLMo-2-1B:

```bash
python main.py \
  run_name=kromhc_olmo2_1b \
  method.selected_method=kromhc \
  model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B \
  train.max_train_steps=20000
```

Train LoRA:

```bash
python main.py \
  run_name=lora_olmo2_1b \
  method.selected_method=lora \
  model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B \
  train.learning_rate=1e-4 \
  train.max_train_steps=20000
```

Train KromHC combined with LoRA:

```bash
python main.py \
  run_name=kromhc_lora_olmo2_1b \
  method.selected_method=kromhc_lora \
  model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B \
  train.use_dual_lr=true \
  train.hc_learning_rate=3e-3 \
  train.adapter_learning_rate=1e-4 \
  train.max_train_steps=20000
```

## Supported methods

mHC variants:

| Method               | Implementation              |
| -------------------- | --------------------------- |
| static Sinkhorn mHC  | `models/static_mHC.py`      |
| dynamic Sinkhorn mHC | `models/dynamic_mHC.py`     |
| static mHC-lite      | `models/static_mHC_lite.py` |
| dynamic mHC-lite     | `models/mHC_lite.py`        |
| static KromHC        | `models/static_KromHC.py`   |
| dynamic KromHC       | `models/KromHC.py`          |
| Delta-KromHC         | `models/delta_KromHC.py`    |

The mHC-lite code is adapted from the public mHC-lite repository: [https://github.com/FFTYYY/mhc-lite](https://github.com/FFTYYY/mhc-lite).
The KromHC code is adapted from the public KromHC repository: [https://github.com/wz1119/KromHC](https://github.com/wz1119/KromHC).

Baselines are implemented through Hugging Face PEFT:

* LoRA
* VeRA
* IA³
* prompt tuning
* layer tuning

The main paper uses OLMo-2. The repository also contains a working OLMo-3 wrapper in `models/model_OLMo_3.py`, but OLMo-3 is not used for the paper experiments.

## Repository structure

| Path                          | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `main.py`                     | Hydra entry point for training and benchmarking              |
| `configs/config.yaml`         | Main configuration file                                      |
| `run_finetuning.py`           | Training and evaluation orchestration                        |
| `train.py`                    | Hugging Face Trainer setup and checkpoint saving             |
| `benchmark.py`                | `lm-evaluation-harness` integration                          |
| `run_benchmarks.py`           | Benchmark orchestration                                      |
| `evaluate_model_on_splits.py` | Held-out validation/test perplexity evaluation               |
| `models/`                     | mHC modules, OLMo wrappers, PEFT injection, and reload logic |
| `data/`                       | Dataset loading, preprocessing, packing, and cache creation  |
| `jobs/`                       | Example cluster job scripts                                  |

## Data

The default dataset is:

```text
allenai/tulu-3-sft-olmo-2-mixture
```

The main config supports either direct Hugging Face loading or a pre-tokenized dataset cache. For large runs, precomputing a tokenized cache is recommended.

Example cache-building scripts are in `data/`:

```bash
python data/build_tokenized_cache.py
```

The default sequence length is 2048, with packed examples and deterministic validation/test splits.

## Training outputs

Training writes a final reloadable model directory. Important files:

| File                    | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `trainable_params.pt`   | Trainable parameters only                        |
| `reload_metadata.json`  | Metadata needed to reconstruct the method        |
| `training_summary.json` | Training/evaluation metrics and parameter counts |
| `resolved_config.yaml`  | Fully resolved Hydra config                      |

Only trainable parameters are saved. The frozen base model is reloaded from Hugging Face.

## Evaluation

Evaluate validation/test perplexity for a saved checkpoint:

```bash
python evaluate_model_on_splits.py \
  --checkpoint_dir path/to/final_model \
  --splits validation test
```

Alternatively, evaluation can be enabled during training through the Hydra config.

## Benchmarking

Benchmarking uses `lm-evaluation-harness`.

Run the configured benchmark suite on a saved checkpoint:

```bash
python main.py \
  mode=benchmark \
  method.selected_method=kromhc \
  benchmark.checkpoint_path=path/to/final_model \
  benchmark.batch_size=auto
```

Run a single benchmark task:

```bash
python main.py \
  mode=benchmark \
  method.selected_method=kromhc \
  benchmark.checkpoint_path=path/to/final_model \
  benchmark.batch_size=auto \
  ~benchmark.tasks \
  ++benchmark.tasks.mmlu.fewshot=5 \
  ++benchmark.tasks.mmlu.metric=acc
```

The default benchmark suite is:

| Task           | Few-shot | Metric              |
| -------------- | -------: | ------------------- |
| BBH            |        3 | exact match         |
| DROP           |        3 | F1                  |
| GSM8K          |        8 | exact match         |
| HellaSwag      |       10 | normalized accuracy |
| Hendrycks MATH |        4 | exact match         |
| MMLU           |        5 | accuracy            |
| PIQA           |        0 | accuracy            |
| TriviaQA       |        5 | exact match         |

## Configuration

All main settings are controlled by Hydra through `configs/config.yaml`. Common overrides:

```bash
# choose method
python main.py method.selected_method=kromhc
python main.py method.selected_method=static_kromhc
python main.py method.selected_method=lora
python main.py method.selected_method=kromhc_lora

# choose model
python main.py model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B
python main.py model.pretrained_model_name_or_path=allenai/OLMo-2-1124-7B

# change number of residual streams
python main.py method.shc_num_streams=2
python main.py method.shc_num_streams=16

# use identity residual routing
python main.py method.shc_ablation_mapping='[res]'

# use separate learning rates for mHC and adapter parameters
python main.py train.use_dual_lr=true train.hc_learning_rate=3e-3 train.adapter_learning_rate=1e-4
```

## License

This repository is released under the **Creative Commons Attribution License (CC BY)**.

## Contact

Maintainer: Floris de Kam; email: [floris.de.kam@student.uva.nl](mailto:floris.de.kam@student.uva.nl)