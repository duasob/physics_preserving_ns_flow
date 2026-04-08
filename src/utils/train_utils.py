from pathlib import Path

import torch
import yaml


def build_optimizer(model, name, lr, weight_decay=None):
    name = name.lower()
    kwargs = {"lr": lr}
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def forward_with_optional_aux(model, coords, fx, use_aux: bool):
    if use_aux:
        out = model(coords, fx, return_aux=True)
        if isinstance(out, tuple) and len(out) == 3:
            pred, pred_base, corr = out
            return {
                "pred": pred,
                "pred_mean": float(pred.mean().item()),
                "pred_base_mean": float(pred_base.mean().item()),
                "corr_mean": float(corr.mean().item()),
            }
        pred = out

    else:
        pred = model(coords, fx)
    return {
        "pred": pred,
        "pred_mean": float(pred.mean().item()),
        "pred_base_mean": None,
        "corr_mean": None,
    }


def ensure_save_dir(cfg: dict) -> Path:
    save_dir = Path(cfg["path"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def write_yaml_config(path: str | Path, payload: dict) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out_path
