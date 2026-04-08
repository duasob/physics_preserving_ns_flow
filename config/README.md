## Config Guide
Due to the modularity of this project, we have a layered config structure:
1. A top-level run config such as [galerkin_latent_cfg.yaml](/Users/bruno/Documents/Y4/FYP/hc_fluid/config/galerkin_latent_cfg.yaml)
2. A backbone default config in [model_config/](/Users/bruno/Documents/Y4/FYP/hc_fluid/config/model_config)
3. Optionally, a hard-constraint config in [hc_config/](/Users/bruno/Documents/Y4/FYP/hc_fluid/config/hc_config) 

The top-level run config is the entrypoint you pass to `main.py` or `test.py`.

### Top-Level Structure

A typical config has these sections:

```yaml
model:
path:
data:
hyper_parameters:
wandb_logging:
optuna:        # only for Optuna search configs
```

### `model`

This section defines the backbone and optional hard constraint wrapper.

Example:

```yaml
model:
  backbone: "Galerkin_Transformer"
  config: "config/model_config/Galerkin_Transformer.yaml"
  hc: "config/hc_config/mean_correction_cfg.yaml"
  args:
    n_hidden: 256
    n_layers: 6
    n_heads: 8
    mlp_ratio: 2
    dropout: 0.15557814394885827
  hc_overrides:
    mode: "latent_head"
    latent_module: "blocks.-1.ln_3"
    latent_dim: 256
    correction_layers: 2
    correction_hidden: 192
    correction_act: "tanh"
    freeze_base: false
```

Fields:

- `backbone`: model name expected by Neural-Solver-Library.
- `config`: backbone default config file in `config/model_config/`.
- `hc`: optional hard-constraint config file in `config/hc_config/`.
- `args`: explicit overrides on top of the backbone default config.
- `hc_overrides`: explicit overrides on top of the HC config.

Notes:

- `model.args` wins over values from `model.config`.
- `model.hc_overrides` wins over values from `model.hc`.
- If `hc` is omitted, the backbone is trained without the mean-correction wrapper.

### `path`

This section controls dataset and checkpoint locations.

```yaml
path:
  root_dir: "data/NavierStokes_V1e-5_N1200_T20"
  save_dir: "checkpoints/NavierStokes_V1e-5_N1200_T20/optuna_1"
```

Fields:

- `root_dir`: either the `.mat` file itself, or the directory containing `NavierStokes_V1e-5_N1200_T20.mat`.
- `save_dir`: output directory for checkpoints, `input_config.yaml`, and `resolved_config.yaml`.

### `data`

This section controls the Navier-Stokes `.mat` loader.

```yaml
data:
  ns_mode: "autoregressive"
  downsamplex: 1
  downsampley: 1
  ntrain: 1000
  ntest: 200
  T_in: 10
  T_out: 10
```

Fields:

- `ns_mode`: `autoregressive` or `single_step`
- `downsamplex`, `downsampley`: spatial subsampling factors
- `ntrain`: number of training samples taken from the start of the dataset
- `ntest`: number of held-out test samples taken from the end of the dataset
- `T_in`: number of input time steps
- `T_out`: number of predicted time steps

Important:

- Training uses only the training split plus an internal validation split.
- The held-out test split is not used during training.
- Testing is done separately with `test.py`.

### Runtime Overrides

Some backbone fields are computed at runtime from the dataset and should be treated as derived values, not as the main source of truth in the YAML:

- `shapelist`
- `task`
- `T_in`
- `T_out`
- `out_dim`
- `fun_dim`

For example, in autoregressive NS with `T_in=10` and `out_dim=1`, the effective `fun_dim` becomes `10` at runtime.

### `hyper_parameters`

This section controls optimization, validation, and early stopping.

Example:

```yaml
hyper_parameters:
  num_epochs: 100
  batch_size: 4
  optimizer: "adamw"
  max_steps: null
  learning_rate: 0.0023381056322343496
  weight_decay: 1e-5
  scheduler: "OneCycleLR"
  pct_start: 0.3
  step_size: 100
  gamma: 0.5
  max_grad_norm: null
  teacher_forcing: 1
  val_size: 100
  val_split_seed: 42
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.0
    monitor: "val_rel_l2"
```

Fields:

- `num_epochs`: total target epoch count
- `batch_size`: train and validation batch size
- `optimizer`: `adam`, `adamw`, or `sgd`
- `max_steps`: optional cap on training batches per epoch
- `learning_rate`, `weight_decay`
- `scheduler`: `none`, `OneCycleLR`, `CosineAnnealingLR`, or `StepLR`
- `pct_start`, `step_size`, `gamma`: scheduler-specific settings
- `max_grad_norm`: optional gradient clipping
- `teacher_forcing`: only relevant for autoregressive training
- `val_size`: validation subset size sampled from the training split
- `val_split_seed`: split seed for reproducibility

Early stopping monitors currently supported:

- `val_rel_l2`
- `val_mse`
- `val_step_rel_l2`

### `wandb_logging`

This section controls Weights & Biases logging.

```yaml
wandb_logging:
  wandb: true
  project: "hc_fluid"
  run_name: "gt_latent_head_1"
  log_every: 100
  image_log_every: 10
```

Fields:

- `wandb`: enable or disable W&B
- `project`: W&B project name
- `run_name`: run name
- `log_every`: scalar log interval in steps
- `image_log_every`: validation image log interval in epochs

### `optuna`

Only Optuna configs need this section.

Example:

```yaml
optuna:
  num_trials: 20
  direction: "minimize"
  save_dir: "checkpoints/NavierStokes_V1e-5_N1200_T20/optuna_gt_latent_head"
  run_name: "gt_latent_head"
  wandb_logging:
    wandb: false
  search_space:
    hyper_parameters.learning_rate:
      type: "float"
      low: 1e-6
      high: 5e-3
      log: true
```

Fields:

- `num_trials`: number of Optuna trials
- `direction`: optimization direction
- `save_dir`: parent directory for per-trial outputs
- `run_name`: prefix for per-trial W&B names
- `wandb_logging`: trial-level W&B overrides
- `search_space`: dotted-path overrides sampled by Optuna
