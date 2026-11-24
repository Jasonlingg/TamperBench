# Configuration Reference

This guide covers how configurations are structured and how to customize them.

These configs are primarily designed to support the benchmark scripts (`benchmark_grid.py` and `optuna_single.py`) for exhaustive benchmarking. They map directly to the dataclass configs used by each attack/evaluation class.

<!-- TODO: Consider migrating to Hydra for more flexible config composition -->

## Config File Location

Default attack configurations live in `configs/whitebox/attacks/`. This directory can be copied and customized as needed—use `--configs-dir` to point scripts to your custom configs.

```text
configs/whitebox/attacks/<attack_name>/
├── grid.yaml                    # Required for benchmark_grid.py
└── single_objective_sweep.yaml  # Required for optuna_single.py
```

**Minimum requirements for scripts:**

- `grid.yaml` with at least a `base` key defining default parameters
- `single_objective_sweep.yaml` if using `optuna_single.py` for sweeps
- Attack subdirectory names must match `AttackName` enum values

## Naming Conventions

Config directory names must match `AttackName` enum values exactly. Evaluation names in `evals:` lists must match `EvalName` enum values. See `src/tamperbench/whitebox/utils/names.py` for all valid names.

## grid.yaml Structure

Used by `benchmark_grid.py` for static benchmark runs.

```yaml
base: &base_cfg
    model_config:
        template: plain
        max_generation_length: 1024
        inference_batch_size: 16
    evals: [strong_reject, mmlu_pro_val]
    per_device_train_batch_size: 8
    learning_rate: 0.0001
    num_train_epochs: 1
    lr_scheduler_type: constant
    optim: adamw_torch
    lora_rank: 16
    max_steps: -1

# Additional variants inherit from base
high_lr:
    <<: *base_cfg
    learning_rate: 0.001

low_rank:
    <<: *base_cfg
    lora_rank: 8
```

### Config to Dataclass Mapping

YAML fields map directly to the attack's config dataclass. For example, with `LoraFinetuneConfig`:

```python
# LoraFinetuneConfig dataclass fields:
@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    lora_rank: int
    # Inherits: learning_rate, per_device_train_batch_size, max_steps, ...
```

```yaml
# grid.yaml - fields map 1:1 to dataclass attributes
base:
    lora_rank: 16
    learning_rate: 0.0001
    per_device_train_batch_size: 8
    # ...
```

See individual config dataclasses in `src/tamperbench/whitebox/attacks/<attack>/` for all available parameters.

### Optimizer Selection

The `optim` field already selects the fine-tuning optimizer. Existing configs use
`adamw_torch`; they keep their current behavior. Both `benchmark_grid.py` and
`optuna_single.py` accept an optional `--optim` override:

```bash
python scripts/whitebox/benchmark_grid.py /path/to/checkpoint \
    --attacks full_parameter_finetune --optim adamw_torch \
    --model-alias adamw_baseline
```

Omitting the flag preserves the YAML selection. For Optuna, an explicit flag also
removes `optim` from the search space so the sweep cannot override it. The selected
optimizer is part of the config passed to the runner. Use separate model aliases
or results directories for optimizer comparisons: existing checkpoint/result
caching does not invalidate previous runs when an optimizer changes.

#### Muon for Full-Parameter Fine-Tuning

`optim: muon` selects native `torch.optim.Muon` with auxiliary AdamW for
`full_parameter_finetune` and `benign_full_parameter_finetune`. Other optimizers
continue to use the existing SFTTrainer optimizer setup. LoRA rejects Muon.

```yaml
optim: muon
learning_rate: 0.00001       # Illustrative; tune independently from the AdamW baseline.
weight_decay: 0.0
muon_momentum: 0.95
muon_nesterov: true
muon_ns_steps: 5
muon_adjust_lr_fn: match_rms_adamw
adamw_learning_rate: null   # Share learning_rate unless explicitly overridden.
```

Weights of hidden `torch.nn.Linear` and Hugging Face `Conv1D` modules go to Muon.
Embedding modules, the output head, biases, normalization weights, and other
parameters go to AdamW. Embedding/output identities take precedence when weights
are tied; frozen parameters are excluded and shared weights are included once.
Unrecognized module types conservatively use AdamW. Models without any eligible
hidden matrix raise an error. The default
`match_rms_adamw` scaling follows the RMS-matching approach in
[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982).
The alternate scaling choice is `original`. Both optimizers use the trainer's
existing weight-decay exclusions, including biases and normalization parameters.
This differs from the paper's treatment of normalization weights; this integration
does not claim to reproduce its training procedure.

The optimizer is constructed after SFTTrainer places the model on its training
device. A combined optimizer exposes both algorithms to the trainer's scheduler,
gradient clearing, and optimizer-state serialization. An explicit auxiliary
AdamW learning rate retains its ratio under the scheduler. Because the pinned
Transformers optimizer enum lacks Muon, SFTConfig uses the `adamw_torch` enum
value while the worker installs the actual combined optimizer before training.
The experiment's `optim: muon` field records the selected algorithm.

A small grid example lives in
`configs/whitebox/attacks_muon/full_parameter_finetune/grid.yaml`. To select it:

```bash
python scripts/whitebox/benchmark_grid.py /path/to/checkpoint \
    --attacks full_parameter_finetune \
    --configs-dir configs/whitebox/attacks_muon --optim muon \
    --model-alias muon_pilot
```

This example is a small pilot configuration, not a tuned research experiment or
an Optuna sweep directory. The existing attack loader uses BF16 model weights;
use a compatible CUDA GPU for realistic experiments. Tiny CPU runs have validated
the optimizer and training worker, but CUDA performance and safety/capability
comparisons remain to be measured. Full fine-tuning uses the same CPU selection
as LoRA when CUDA is unavailable.

This integration targets unsharded training. FSDP and DeepSpeed are rejected;
distributed execution has not been validated. The optimizer can serialize and
restore its state, but the attack runner still saves only the final model with
`save_strategy="no"`; automatic interrupted-attack recovery is not added here.

### Evaluations

The `evals` field specifies which evaluations to run after the attack completes:

```yaml
evals: [strong_reject, mmlu_pro_val]
```

Values must match `EvalName` enum (see Evaluation Name Mapping above). Multiple evals can be specified.

## single_objective_sweep.yaml Structure

Used by `optuna_single.py` for hyperparameter optimization.

```yaml
evals: [strong_reject, mmlu_pro_val]
sweep:
  per_device_train_batch_size:
    choices: [8, 16, 32, 64]

  learning_rate:
    type: "float"
    low: 1.0e-6
    high: 1.0e-3
    log: true

  max_steps:
    choices: [16, 64, 128, 256, 512]

  lr_scheduler_type:
    choices: [constant, cosine]

  model_config.template:
    choices: [plain, instruction_response, generic_chat]

  lora_rank:
    type: categorical
    choices: [8, 16, 32, 64]
```

### Parameter Types

**Categorical (choices):**

```yaml
lr_scheduler_type:
  choices: [constant, cosine, linear]
```

**Float range:**

```yaml
learning_rate:
  type: "float"
  low: 1.0e-6
  high: 1.0e-3
  log: true    # Log-uniform sampling
```

**Integer range:**

```yaml
num_layers:
  type: "int"
  low: 1
  high: 12
  step: 1      # Optional step size
```

**Explicit categorical:**

```yaml
lora_rank:
  type: categorical
  choices: [8, 16, 32, 64]
```

### Nested Parameters

Use dotted notation for nested config values:

```yaml
sweep:
  model_config.template:
    choices: [plain, generic_chat]

  model_config.max_generation_length:
    choices: [512, 1024, 2048]
```

## Model Config Templates

The `template` field in `model_config` controls prompt formatting.

 | Template               | Description                                                      |
 | ---------------------- | ---------------------------------------------------------------- |
 | `native`               | Use the tokenizer's built-in chat template                       |
 | `plain`                | No special formatting                                            |
 | `generic_chat`         | Standard chat format with user/assistant turns                   |
 | `instruction_response` | Instruction/response pairs                                       |

Templates are defined in `src/tamperbench/whitebox/utils/models/templates.py`.

### Custom Templates

Add new templates by extending `TEMPLATE_REGISTRY` in `templates.py`:

```python
# In templates.py
TEMPLATE_REGISTRY[TemplateName.MY_TEMPLATE] = TextTemplate(
    user_prefix="<|user|>\n",
    assistant_prefix="<|assistant|>\n",
    end_turn="<|end|>",
)
```

## When to Edit Which File

 | Goal                               | File to Edit                  |
 | ---------------------------------- | ----------------------------- |
 | Run benchmarks with fixed params   | `grid.yaml`                   |
 | Add new config variant             | `grid.yaml` (new key)         |
 | Change hyperparameter search space | `single_objective_sweep.yaml` |
 | New attack type                    | Create new directory          |

## Creating a New Attack Config

1. Create directory: `configs/whitebox/attacks/<attack_name>/`

2. Add `grid.yaml` with base config:

   ```yaml
   base:
       model_config:
           template: plain
           max_generation_length: 1024
           inference_batch_size: 16
       evals: [strong_reject, mmlu_pro_val]
       # Add attack-specific params
   ```

3. Add `single_objective_sweep.yaml` (if using Optuna sweeps):

   ```yaml
   evals: [strong_reject, mmlu_pro_val]
   sweep:
     # Define search space
   ```

4. Ensure attack subdirectory name matches `AttackName` enum in `names.py`

## Next Steps

- [USAGE.md](USAGE.md) - Running benchmarks with these configs
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Adding new attacks (requires config)
