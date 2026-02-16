from logging import config
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import MindFlowNSLoadConfig, create_mindflow_loaders
import wandb
import sys
import random
import yaml


# Add Neural-Solver-Library to path
repo_root = Path('src/Neural-Solver-Library').resolve()
sys.path.insert(0, str(repo_root))

from models.Galerkin_Transformer import Model as GalerkinTransformer


NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LOG_EVERY = 10  # steps
IMAGE_LOG_EVERY = 1  # epochs
LOG_WANDB = True 
OPTIMIZER = "adam"

@dataclass
class Args:
    unified_pos: bool = False
    geotype: str = 'structured'
    shapelist: tuple = (128, 128)
    ref: int = 1
    fun_dim: int = 1
    space_dim: int = 2
    n_hidden: int = 128
    act: str = 'gelu'
    time_input: bool = False
    n_heads: int = 4
    dropout: float = 0.0
    mlp_ratio: int = 4
    out_dim: int = 1
    n_layers: int = 4
    use_mean_correction: bool = False
    correction_hidden: Optional[int] = None
    correction_layers: int = 0
    correction_act: str = 'gelu'

def make_grid(h, w, device, dtype):
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)
    return grid.view(-1, 2)

def build_optimizer(model, name, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")

def prepare_batch(batch, grid, device="cuda"):
    x = batch["x"].to(device).squeeze(-1).squeeze(-1)  # (B, H, W)
    y = batch["y"].to(device).squeeze(-1).squeeze(-1)  # (B, H, W)
    # Test data can include extra time-like dimensions; use a single-step slice for now.
    while x.dim() > 3:
        x = x[:, 0]
        y = y[:, 0]
    if x.dim() != 3:
        raise ValueError(f"Expected x to have 3 dims (B,H,W) after slicing, got shape {tuple(x.shape)}")
    b, h, w = x.shape
    # grid should be (H, W, 2) or (H*W, 2); make it (H*W, 2)
    if grid.dim() == 3:
        grid_flat = grid.view(h * w, -1)
    else:
        grid_flat = grid

    coords = grid_flat.to(device).unsqueeze(0).repeat(b, 1, 1)  # (B, H*W, 2)
    fx = x.view(b, h * w, 1)  # (B, H*W, 1)
    target = y.view(b, h * w, 1)
    return coords, fx, target

def _to_rgb(image, cmap_name, vmin=None, vmax=None):
    img = image.detach().cpu().numpy()
    if vmin is None:
        vmin = float(img.min())
    if vmax is None:
        vmax = float(img.max())
    if vmax == vmin:
        vmax = vmin + 1e-12
    img_norm = (img - vmin) / (vmax - vmin)
    cmap = plt.colormaps[cmap_name]
    rgba = cmap(img_norm)
    return (rgba[..., :3] * 255).astype("uint8")


def log_prediction_images(pred, target, fx, h, w, *, prefix, epoch):
    input_img = fx[0, :, 0].view(h, w)
    pred_img = pred[0, :, 0].view(h, w)
    target_img = target[0, :, 0].view(h, w)
    diff_img = (pred_img - target_img).abs()
    pred_merge = torch.cat([input_img, pred_img], dim=1)
    target_merge = torch.cat([input_img, target_img], dim=1)
    vmin = min(float(input_img.min()), float(target_img.min()), float(pred_img.min()))
    vmax = max(float(input_img.max()), float(target_img.max()), float(pred_img.max()))
    pred_rgb = _to_rgb(pred_merge, "viridis", vmin=vmin, vmax=vmax)
    target_rgb = _to_rgb(target_merge, "viridis", vmin=vmin, vmax=vmax)
    diff_max = float(diff_img.max())
    diff_rgb = _to_rgb(diff_img, "bwr", vmin=-diff_max, vmax=diff_max)

    wandb.log(
        {
            f"{prefix}/pred": wandb.Image(pred_rgb),
            f"{prefix}/target": wandb.Image(target_rgb),
            f"{prefix}/abs_error": wandb.Image(diff_rgb),
            "epoch": epoch + 1,
        }
    )

def setup_hc(cfg):
    if cfg["model"]["hc"] is not None:
        with open(f"config/hc_config/{cfg["model"]["hc"]}_cfg.yaml", "r") as f:
            hc_cfg = yaml.safe_load(f)
        # TODO: from here, init the HC module
        return hc_cfg
    else:
        return None
def setup(cfg):
    data_root = Path()
    wandb_cfg = cfg["wandb_logging"] 
    if wandb_cfg["wandb"]:
        wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg["run_name"],
            config=cfg,
        )

    data_config = MindFlowNSLoadConfig(
        root=Path(cfg["path"]["root_dir"]),
        batch_size=cfg["hyper_parameters"]["batch_size"],
        test_batch_size=cfg["hyper_parameters"]["batch_size"],
        num_workers=0,
    )

    train_loader, test_loader = create_mindflow_loaders(data_config)

    mkdir_path = Path(cfg["path"]["save_dir"])
    mkdir_path.mkdir(parents=True, exist_ok=True)


    hc_cfg = setup_hc(cfg)
    
    args = Args()
    # TODO: replace this with a modular structure
    if hc_cfg is not None:
        args.use_mean_correction = True 
        args.correction_hidden = hc_cfg["correction_hidden"]
        args.correction_layers = hc_cfg["correction_layers"]
        args.correction_act = hc_cfg["correction_act"]

    return args, train_loader, test_loader


def train(cfg=None, from_checkpoint=None):
    if cfg is None:
        raise ValueError("Please provide a config file")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, train_loader, test_loader = setup(cfg)

    if from_checkpoint is not None:
        model =  (args).to(device)
        model.load_state_dict(torch.load(from_checkpoint, map_location=device))
        print(f"Loaded model from checkpoint: {from_checkpoint}")
    else:
        model = GalerkinTransformer(args).to(device)

    criterion = nn.MSELoss()

    first_batch = next(iter(train_loader))
    h, w = first_batch['x'].shape[1], first_batch['x'].shape[2]
    grid = make_grid(h, w, device=device, dtype=first_batch['x'].dtype)

    model.train()

    train_cfg = cfg["hyper_parameters"]
    wandb_cfg = cfg["wandb_logging"]

    optimizer = build_optimizer(model, train_cfg["optimizer"], train_cfg["learning_rate"])
    
    if train_cfg["max_steps"] is not None:
        max_len = min(len(test_loader) - 1, train_cfg["max_steps"])
    else:
        max_len = len(test_loader) - 1
    test_epoch_loss = None
    for epoch in range(train_cfg["num_epochs"]):
        total_loss = 0.0 #config
        for step, batch in enumerate(train_loader):
            coords, fx, target = prepare_batch(batch, grid, device=device)
            if args.use_mean_correction:
                pred, pred_base, corr = model(coords, fx, return_aux=True)
                pred_mean = float(pred.mean().item())
                pred_base_mean = float(pred_base.mean().item())
                corr_mean = float(corr.mean().item())
            else:
                pred = model(coords, fx)
                pred_mean = float(pred.mean().item())
                pred_base_mean = None
                corr_mean = None
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % wandb_cfg["log_every"] == 0 and wandb_cfg["wandb"]:
                global_step = epoch * len(train_loader) + step + 1
                log_payload = {
                    "train/step_loss": loss.item(),
                    "train/pred_mean": pred_mean,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                }
                if pred_base_mean is not None:
                    log_payload["train/pred_base_mean"] = pred_base_mean
                if corr_mean is not None:
                    log_payload["train/corr_mean"] = corr_mean
                wandb.log(log_payload)
            if (step + 1) % wandb_cfg["log_every"] == 0:
                msg = f"step {step + 1}: pred_mean={pred_mean:.6f}"
                if pred_base_mean is not None:
                    msg += f", pred_base_mean={pred_base_mean:.6f}, corr_mean={corr_mean:.6f}"
                print(msg)
            if train_cfg["max_steps"] is not None and (step + 1) >= train_cfg["max_steps"]:
                break
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}: loss={avg_loss:.6f}')
        torch.save(model.state_dict(), f'{cfg["path"]["save_dir"]}/galerkin_transformer_epoch{epoch + 1}.pth')
        if wandb_cfg["wandb"]:
            wandb.log({"epoch": epoch + 1, "train/epoch_loss": avg_loss})
        
        
        # Eval 
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_step, batch in enumerate(test_loader):
                coords, fx, target = prepare_batch(batch, grid, device=device)
                if args.use_mean_correction:
                    pred, pred_base, corr = model(coords, fx, return_aux=True)
                    pred_mean = float(pred.mean().item())
                    pred_base_mean = float(pred_base.mean().item())
                    corr_mean = float(corr.mean().item())
                else:
                    pred = model(coords, fx)
                    pred_mean = float(pred.mean().item())
                    pred_base_mean = None
                    corr_mean = None
                loss = criterion(pred, target)
                test_loss += loss.item()
                if (
                    wandb_cfg["wandb"]
                    and epoch % wandb_cfg["image_log_every"] == 0
                    and test_step == random.randint(0, max_len) # Log a random batch each epoch
                ):
                    log_prediction_images(pred, target, fx, h, w, prefix="inference", epoch=epoch)
                if wandb_cfg["wandb"] and (test_step + 1) % wandb_cfg["log_every"] == 0:
                    log_payload = {
                        "test/pred_mean": pred_mean,
                        "epoch": epoch + 1,
                    }
                    if pred_base_mean is not None:
                        log_payload["test/pred_base_mean"] = pred_base_mean
                    if corr_mean is not None:
                        log_payload["test/corr_mean"] = corr_mean
                    wandb.log(log_payload)
        test_epoch_loss = test_loss / len(test_loader)
        if wandb_cfg["wandb"]:
            wandb.log({"epoch": epoch + 1, "test/epoch_loss": test_epoch_loss})
        model.train()
    return test_epoch_loss

def _suggest_value(trial, name, spec):
    if isinstance(spec, dict):
        spec_type = spec.get("type", "").lower()
        if spec_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        if spec_type == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if spec_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        raise ValueError(f"Unknown optuna spec type for {name}: {spec_type}")
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    raise ValueError(f"Invalid optuna spec for {name}: {spec}")


def _apply_optuna_search_space(cfg, trial, search_space):
    trial_cfg = deepcopy(cfg)
    hp = trial_cfg.setdefault("hyper_parameters", {})
    for name, spec in search_space.items():
        hp[name] = _suggest_value(trial, name, spec)
    return trial_cfg


def optuna_search(cfg=None):
    import optuna

    if cfg is None:
        raise ValueError("Please provide a config file")

    optuna_cfg = cfg.get("optuna", {})
    num_trials = int(optuna_cfg.get("num_trials", 20))
    direction = optuna_cfg.get("direction", "minimize")
    search_space = optuna_cfg.get("search_space", {})
    if not search_space:
        raise ValueError("optuna.search_space is required")

    save_base = optuna_cfg.get("save_dir", cfg["path"]["save_dir"])
    run_name_base = optuna_cfg.get("run_name", "garlekin_fno2d_optuna")
    wandb_overrides = optuna_cfg.get("wandb_logging", {"wandb": False})

    trial_num = 0

    def objective(trial):
        nonlocal trial_num
        trial_num += 1
        trial_cfg = _apply_optuna_search_space(cfg, trial, search_space)
        trial_cfg = deepcopy(trial_cfg)
        
        # Override other config
        trial_cfg["path"]["save_dir"] = f"{save_base}/optuna_{trial_num}"
        trial_cfg["wandb_logging"] = {
            **trial_cfg.get("wandb_logging", {}),
            **wandb_overrides,
            "run_name": f"{run_name_base}_{trial_num}",
        }
        return train(cfg=trial_cfg, from_checkpoint=None)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=num_trials)
    return study
                    
