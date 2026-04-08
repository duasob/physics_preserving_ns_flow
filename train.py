import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.data_utils import make_grid, prepare_batch
from src.utils.logging_utils import (
    finish_wandb_if_active,
    init_wandb_if_enabled,
    log_prediction_images,
)
from src.utils.modeling import checkpoint_prefix, create_model
from src.utils.optuna_utils import apply_optuna_search_space
from src.utils.train_utils import (
    build_optimizer,
    ensure_save_dir,
    forward_with_optional_aux,
    write_yaml_config,
)


class _DictTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _resolve_ns_mat_file(root_dir: str | Path) -> Path | None:
    root = Path(root_dir)
    if root.is_file() and root.suffix == ".mat":
        return root

    candidate = root / "NavierStokes_V1e-5_N1200_T20.mat"
    if candidate.exists():
        return candidate

    return None


def _create_nsl_ns_mat_loaders(cfg: dict):
    mat_path = _resolve_ns_mat_file(cfg["path"]["root_dir"])
    if mat_path is None:
        return None

    data_cfg = cfg.get("data", {})
    model_args = cfg.get("model", {}).get("args", {})
    hp_cfg = cfg["hyper_parameters"]

    try:
        import scipy.io as scio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to load Navier-Stokes .mat datasets. "
            "Install it with `pip install scipy`."
        ) from exc

    raw = scio.loadmat(str(mat_path))
    if "u" not in raw:
        raise KeyError(
            f"Expected key 'u' in {mat_path}, found keys: {sorted(raw.keys())}"
        )
    u = raw["u"]
    if u.ndim != 4:
        raise ValueError(f"Expected 'u' with 4 dims (N,H,W,T), got shape {u.shape}")

    n_samples, h_full, w_full, t_full = u.shape
    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    t_in = _as_int(data_cfg.get("T_in", model_args.get("T_in")), 10)
    t_out = _as_int(data_cfg.get("T_out", model_args.get("T_out")), 10)
    out_dim = _as_int(data_cfg.get("out_dim", model_args.get("out_dim")), 1)
    ntrain = _as_int(data_cfg.get("ntrain", model_args.get("ntrain")), 1000)
    ns_mode = str(data_cfg.get("ns_mode", "autoregressive")).strip().lower()
    if ns_mode not in {"autoregressive", "single_step"}:
        raise ValueError("data.ns_mode must be one of: autoregressive, single_step")

    if t_full < 2:
        raise ValueError(f"Need at least 2 time steps, got {t_full}")

    ntrain = min(ntrain, n_samples)
    if ntrain <= 0 or ntrain >= n_samples:
        raise ValueError(
            f"Invalid training split: n_samples={n_samples}, ntrain={ntrain}"
        )

    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    train_u = u[:ntrain, ::r1, ::r2][:, :s1, :s2]
    if ns_mode == "single_step":
        input_t = min(max(t_in - 1, 0), t_full - 2)
        target_t = input_t + 1
        x_train = (
            torch.from_numpy(train_u[..., input_t]).float().unsqueeze(-1).unsqueeze(-1)
        )
        y_train = (
            torch.from_numpy(train_u[..., target_t]).float().unsqueeze(-1).unsqueeze(-1)
        )
        time_desc = f"time_pair=({input_t}->{target_t})"
        fun_dim = int(out_dim)
    else:
        if t_in <= 0 or t_out <= 0:
            raise ValueError(f"T_in and T_out must be positive, got {t_in}, {t_out}")
        if out_dim != 1:
            raise ValueError(
                "NS .mat loader currently supports out_dim=1. "
                f"Received out_dim={out_dim}."
            )
        if t_in + t_out > t_full:
            raise ValueError(
                f"T_in + T_out exceeds available time dimension: "
                f"{t_in} + {t_out} > {t_full}"
            )

        train_a = train_u[..., None, :t_in]
        train_y = train_u[..., None, t_in : t_in + t_out]
        x_train = torch.from_numpy(
            train_a.reshape(train_a.shape[0], -1, out_dim * t_in)
        ).float()
        y_train = torch.from_numpy(
            train_y.reshape(train_y.shape[0], -1, out_dim * t_out)
        ).float()
        time_desc = f"time_window=([0:{t_in}] -> [{t_in}:{t_in + t_out}])"
        fun_dim = int(out_dim * t_in)

    train_ds = _DictTensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    ns_meta = {
        "ns_mode": ns_mode,
        "shapelist": (int(s1), int(s2)),
        "t_in": int(t_in),
        "t_out": 1 if ns_mode == "single_step" else int(t_out),
        "out_dim": int(out_dim),
        "fun_dim": int(fun_dim),
    }
    train_loader.ns_meta = ns_meta

    print(
        "Using NS .mat loader:",
        f"path={mat_path}",
        f"shape={u.shape}",
        f"downsample=({r1},{r2})",
        f"mode={ns_mode}",
        time_desc,
    )
    print(
        "Prepared tensors:",
        f"train={tuple(x_train.shape)}->{tuple(y_train.shape)}",
    )
    return train_loader


def _create_nsl_ns_mat_test_loader(
    cfg: dict,
    *,
    root_dir: str | Path | None = None,
    batch_size: int | None = None,
):
    mat_path = _resolve_ns_mat_file(root_dir or cfg["path"]["root_dir"])
    if mat_path is None:
        return None

    data_cfg = cfg.get("data", {})
    model_args = cfg.get("model", {}).get("args", {})
    hp_cfg = cfg["hyper_parameters"]

    try:
        import scipy.io as scio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to load Navier-Stokes .mat datasets. "
            "Install it with `pip install scipy`."
        ) from exc

    raw = scio.loadmat(str(mat_path))
    if "u" not in raw:
        raise KeyError(
            f"Expected key 'u' in {mat_path}, found keys: {sorted(raw.keys())}"
        )
    u = raw["u"]
    if u.ndim != 4:
        raise ValueError(f"Expected 'u' with 4 dims (N,H,W,T), got shape {u.shape}")

    n_samples, h_full, w_full, t_full = u.shape
    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    t_in = _as_int(data_cfg.get("T_in", model_args.get("T_in")), 10)
    t_out = _as_int(data_cfg.get("T_out", model_args.get("T_out")), 10)
    out_dim = _as_int(data_cfg.get("out_dim", model_args.get("out_dim")), 1)
    ns_mode = str(data_cfg.get("ns_mode", "autoregressive")).strip().lower()
    if ns_mode not in {"autoregressive", "single_step"}:
        raise ValueError("data.ns_mode must be one of: autoregressive, single_step")

    if t_full < 2:
        raise ValueError(f"Need at least 2 time steps, got {t_full}")
    if n_samples <= 0:
        raise ValueError(f"No samples found in test dataset at {mat_path}")

    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    test_u = u[:, ::r1, ::r2][:, :s1, :s2]

    if ns_mode == "single_step":
        input_t = min(max(t_in - 1, 0), t_full - 2)
        target_t = input_t + 1
        x_test = (
            torch.from_numpy(test_u[..., input_t]).float().unsqueeze(-1).unsqueeze(-1)
        )
        y_test = (
            torch.from_numpy(test_u[..., target_t]).float().unsqueeze(-1).unsqueeze(-1)
        )
        time_desc = f"time_pair=({input_t}->{target_t})"
        fun_dim = int(out_dim)
    else:
        if t_in <= 0 or t_out <= 0:
            raise ValueError(f"T_in and T_out must be positive, got {t_in}, {t_out}")
        if out_dim != 1:
            raise ValueError(
                "NS .mat loader currently supports out_dim=1. "
                f"Received out_dim={out_dim}."
            )
        if t_in + t_out > t_full:
            raise ValueError(
                f"T_in + T_out exceeds available time dimension: "
                f"{t_in} + {t_out} > {t_full}"
            )

        test_a = test_u[..., None, :t_in]
        test_y = test_u[..., None, t_in : t_in + t_out]
        x_test = torch.from_numpy(
            test_a.reshape(test_a.shape[0], -1, out_dim * t_in)
        ).float()
        y_test = torch.from_numpy(
            test_y.reshape(test_y.shape[0], -1, out_dim * t_out)
        ).float()
        time_desc = f"time_window=([0:{t_in}] -> [{t_in}:{t_in + t_out}])"
        fun_dim = int(out_dim * t_in)

    test_ds = _DictTensorDataset(x_test, y_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size or hp_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    ns_meta = {
        "ns_mode": ns_mode,
        "shapelist": (int(s1), int(s2)),
        "t_in": int(t_in),
        "t_out": 1 if ns_mode == "single_step" else int(t_out),
        "out_dim": int(out_dim),
        "fun_dim": int(fun_dim),
    }
    test_loader.ns_meta = ns_meta

    print(
        "Using NS .mat test loader:",
        f"path={mat_path}",
        f"shape={u.shape}",
        f"downsample=({r1},{r2})",
        f"mode={ns_mode}",
        time_desc,
    )
    print("Prepared test tensors:", f"test={tuple(x_test.shape)}->{tuple(y_test.shape)}")
    return test_loader


def _build_train_val_loaders(train_loader: DataLoader, cfg: dict):
    val_size_cfg = cfg.get("hyper_parameters", {}).get("val_size", 100)
    val_size = int(val_size_cfg)
    dataset = train_loader.dataset
    total = len(dataset)
    if total < 2:
        raise ValueError(
            "Need at least 2 training samples to create train/validation split."
        )

    val_size = min(max(val_size, 1), total - 1)
    train_size = total - val_size
    seed = int(cfg.get("hyper_parameters", {}).get("val_split_seed", 42))
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader_split = DataLoader(
        train_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
        collate_fn=train_loader.collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=False,
        collate_fn=train_loader.collate_fn,
    )
    if hasattr(train_loader, "ns_meta"):
        train_loader_split.ns_meta = train_loader.ns_meta
        val_loader.ns_meta = train_loader.ns_meta
    return train_loader_split, val_loader


def _relative_l2_per_sample(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
):
    diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
    diff_norm = torch.norm(diff, p=2, dim=1)
    tgt_norm = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1).clamp_min(
        eps
    )
    return diff_norm / tgt_norm


def _relative_l2_sum(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    return _relative_l2_per_sample(pred, target, eps=eps).sum()


def _prepare_batch_autoregressive(batch, grid_flat, device="cuda"):
    fx = batch["x"].to(device)
    target = batch["y"].to(device)

    if fx.dim() == 2:
        fx = fx.unsqueeze(-1)
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    if fx.dim() != 3:
        raise ValueError(f"Expected x to have 3 dims (B,N,C), got {tuple(fx.shape)}")
    if target.dim() != 3:
        raise ValueError(
            f"Expected y to have 3 dims (B,N,C), got {tuple(target.shape)}"
        )

    batch_size = int(fx.shape[0])
    coords = grid_flat.to(device).unsqueeze(0).repeat(batch_size, 1, 1)
    return coords, fx, target


def _rollout_autoregressive(
    model,
    coords,
    fx,
    *,
    use_aux: bool,
    t_out: int,
    out_dim: int,
    target: torch.Tensor | None = None,
    teacher_forcing: bool = False,
):
    if int(t_out) <= 0:
        raise ValueError(f"t_out must be positive, got {t_out}")
    if int(out_dim) <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")

    preds = []
    step_rel_l2 = torch.zeros((), device=coords.device)
    pred_mean_sum = 0.0
    pred_mean_count = 0
    pred_base_sum = 0.0
    pred_base_count = 0
    corr_sum = 0.0
    corr_count = 0

    current_fx = fx
    for t in range(int(t_out)):
        out = forward_with_optional_aux(model, coords, current_fx, use_aux=use_aux)
        pred_t = out["pred"]
        preds.append(pred_t)

        pred_mean_sum += float(out["pred_mean"])
        pred_mean_count += 1
        if out["pred_base_mean"] is not None:
            pred_base_sum += float(out["pred_base_mean"])
            pred_base_count += 1
        if out["corr_mean"] is not None:
            corr_sum += float(out["corr_mean"])
            corr_count += 1

        if target is not None:
            y_t = target[..., out_dim * t : out_dim * (t + 1)]
            step_rel_l2 = step_rel_l2 + _relative_l2_sum(pred_t, y_t)
            next_fx_tail = y_t if teacher_forcing else pred_t
        else:
            next_fx_tail = pred_t

        if current_fx is not None:
            if int(current_fx.shape[-1]) < int(out_dim):
                raise ValueError(
                    f"fx channel dim ({current_fx.shape[-1]}) is smaller than out_dim ({out_dim})"
                )
            current_fx = torch.cat((current_fx[..., out_dim:], next_fx_tail), dim=-1)

    pred = torch.cat(preds, dim=-1)
    stats = {
        "pred_mean": pred_mean_sum / max(pred_mean_count, 1),
        "pred_base_mean": None
        if pred_base_count == 0
        else pred_base_sum / pred_base_count,
        "corr_mean": None if corr_count == 0 else corr_sum / corr_count,
    }
    return pred, step_rel_l2, stats


def _evaluate_loader_autoregressive(
    model,
    loader,
    grid_flat,
    device,
    use_aux,
    t_out: int,
    out_dim: int,
    max_steps: int | None = None,
):
    mse_sum = 0.0
    rel_l2_sum = 0.0
    step_rel_l2_sum = 0.0
    samples = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            coords, fx, target = _prepare_batch_autoregressive(
                batch, grid_flat, device=device
            )
            pred, step_rel_l2, _stats = _rollout_autoregressive(
                model,
                coords,
                fx,
                use_aux=use_aux,
                t_out=t_out,
                out_dim=out_dim,
                target=target,
                teacher_forcing=False,
            )
            batch_size = int(target.shape[0])
            mse_per_sample = F.mse_loss(pred, target, reduction="none").reshape(
                batch_size, -1
            )
            mse_sum += float(mse_per_sample.mean(dim=1).sum().item())
            rel_l2_sum += float(_relative_l2_sum(pred, target).item())
            step_rel_l2_sum += float(step_rel_l2.item())
            samples += batch_size
            if max_steps is not None and (step + 1) >= max_steps:
                break
    denom = max(samples, 1)
    return {
        "mse": mse_sum / denom,
        "rel_l2": rel_l2_sum / denom,
        "step_rel_l2": step_rel_l2_sum / (denom * max(int(t_out), 1)),
    }


def _evaluate_loader(model, loader, grid, device, use_aux, criterion, max_steps=None):
    mse_sum = 0.0
    rel_l2_sum = 0.0
    samples = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            coords, fx, target = prepare_batch(batch, grid, device=device)
            out = forward_with_optional_aux(model, coords, fx, use_aux=use_aux)
            pred = out["pred"]
            loss = criterion(pred, target)
            batch_size = int(target.shape[0])
            mse_sum += loss.item() * batch_size
            rel_l2_sum += float(_relative_l2_per_sample(pred, target).sum().item())
            samples += batch_size
            if max_steps is not None and (step + 1) >= max_steps:
                break
    denom = max(samples, 1)
    return {"mse": mse_sum / denom, "rel_l2": rel_l2_sum / denom}


def _monitor_value(monitor: str, val_metrics: dict):
    if monitor == "val_rel_l2":
        return float(val_metrics["rel_l2"])
    if monitor == "val_mse":
        return float(val_metrics["mse"])
    if monitor == "val_step_rel_l2":
        return float(val_metrics["step_rel_l2"])
    raise ValueError(f"Unknown early stopping monitor: {monitor}")


def setup(cfg):
    init_wandb_if_enabled(cfg)
    loaders = _create_nsl_ns_mat_loaders(cfg)
    if loaders is None:
        raise ValueError(
            "This repository now supports only the Navier-Stokes .mat dataset. "
            "Set path.root_dir to a .mat file or a directory containing "
            "'NavierStokes_V1e-5_N1200_T20.mat'."
        )
    train_loader = loaders
    ensure_save_dir(cfg)
    return train_loader


def _normalize_max_steps(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {"none", "null", ""}:
            return None
        return int(text)
    return int(raw_value)


def _build_scheduler(optimizer, train_cfg: dict, steps_per_epoch: int):
    scheduler_name = str(train_cfg.get("scheduler", "none")).strip()
    scheduler_key = scheduler_name.lower()
    if scheduler_key in {"", "none", "null"}:
        return None, False

    if scheduler_key == "onecyclelr":
        pct_start = float(train_cfg.get("pct_start", 0.3))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(train_cfg["learning_rate"]),
            epochs=int(train_cfg["num_epochs"]),
            steps_per_epoch=max(int(steps_per_epoch), 1),
            pct_start=pct_start,
        )
        return scheduler, True

    if scheduler_key == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg["num_epochs"])
        )
        return scheduler, False

    if scheduler_key == "steplr":
        step_size = int(train_cfg.get("step_size", 100))
        gamma = float(train_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        return scheduler, False

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}. "
        "Use one of: none, OneCycleLR, CosineAnnealingLR, StepLR."
    )


def _resolve_hparam(train_cfg: dict, model_args, key: str, default):
    if key in train_cfg and train_cfg[key] is not None:
        return train_cfg[key]
    if model_args is not None and hasattr(model_args, key):
        value = getattr(model_args, key)
        if value is not None:
            return value
    return default


def _set_global_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass


def _build_resolved_training_config(
    cfg: dict,
    model_args,
    *,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay,
    max_grad_norm,
    teacher_forcing: bool,
    scheduler_cfg: dict,
):
    resolved_cfg = deepcopy(cfg)

    model_section = resolved_cfg.setdefault("model", {})
    args_section = model_section.setdefault("args", {})
    for key, value in vars(model_args).items():
        if key in {"model", "save_name", "eval", "vis_num", "vis_bound"}:
            continue
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        args_section[key] = value

    hp_cfg = resolved_cfg.setdefault("hyper_parameters", {})
    hp_cfg["num_epochs"] = int(num_epochs)
    hp_cfg["learning_rate"] = float(learning_rate)
    hp_cfg["optimizer"] = str(optimizer_name)
    hp_cfg["weight_decay"] = None if weight_decay is None else float(weight_decay)
    hp_cfg["max_grad_norm"] = (
        None if max_grad_norm is None else float(max_grad_norm)
    )
    hp_cfg["teacher_forcing"] = int(bool(teacher_forcing))
    hp_cfg["scheduler"] = scheduler_cfg["scheduler"]
    hp_cfg["pct_start"] = float(scheduler_cfg["pct_start"])
    hp_cfg["step_size"] = int(scheduler_cfg["step_size"])
    hp_cfg["gamma"] = float(scheduler_cfg["gamma"])

    resolved_cfg.pop("optuna", None)
    return resolved_cfg


def _write_run_config_snapshots(
    cfg: dict,
    model_args,
    *,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay,
    max_grad_norm,
    teacher_forcing: bool,
    scheduler_cfg: dict,
):
    save_dir = ensure_save_dir(cfg)
    write_yaml_config(save_dir / "input_config.yaml", deepcopy(cfg))
    resolved_cfg = _build_resolved_training_config(
        cfg,
        model_args,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        teacher_forcing=teacher_forcing,
        scheduler_cfg=scheduler_cfg,
    )
    write_yaml_config(save_dir / "resolved_config.yaml", resolved_cfg)


def _save_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_rel_l2: float,
    best_monitor: float,
    stale_epochs: int,
    metrics: dict | None = None,
):
    payload = {
        "checkpoint_version": 2,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "best_val_rel_l2": float(best_val_rel_l2),
        "best_monitor": float(best_monitor),
        "stale_epochs": int(stale_epochs),
        "metrics": {} if metrics is None else deepcopy(metrics),
    }
    torch.save(payload, path)


def _restore_training_state(
    checkpoint_path: str | Path,
    *,
    device,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        print(
            "Resume requested, but checkpoint only contains model weights. "
            "Optimizer and scheduler will restart from scratch."
        )
        return {
            "start_epoch": 0,
            "best_val_rel_l2": float("inf"),
            "best_monitor": float("inf"),
            "stale_epochs": 0,
        }

    if "optimizer_state_dict" not in checkpoint:
        print(
            "Resume requested, but checkpoint has no optimizer state. "
            "Optimizer and scheduler will restart from scratch."
        )
        return {
            "start_epoch": 0,
            "best_val_rel_l2": float("inf"),
            "best_monitor": float("inf"),
            "stale_epochs": 0,
        }

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif scheduler_state is not None:
        print(
            "Checkpoint contains scheduler state, but the current run does not use "
            "a scheduler. Ignoring saved scheduler state."
        )
    elif scheduler is not None:
        print(
            "Current run uses a scheduler, but the checkpoint has no scheduler state. "
            "The scheduler will restart from scratch."
        )

    start_epoch = max(int(checkpoint.get("epoch", 0)), 0)
    best_val_rel_l2 = float(checkpoint.get("best_val_rel_l2", float("inf")))
    best_monitor = float(checkpoint.get("best_monitor", float("inf")))
    stale_epochs = int(checkpoint.get("stale_epochs", 0))
    current_lr = float(optimizer.param_groups[0]["lr"])
    print(
        f"Resumed training state from checkpoint: {checkpoint_path} "
        f"(next_epoch={start_epoch + 1}, lr={current_lr:.8f})"
    )
    return {
        "start_epoch": start_epoch,
        "best_val_rel_l2": best_val_rel_l2,
        "best_monitor": best_monitor,
        "stale_epochs": stale_epochs,
    }


def train(cfg=None, from_checkpoint=None, resume_checkpoint=None):
    if cfg is None:
        raise ValueError("Please provide a config file")
    if from_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError(
            "Use only one of from_checkpoint or resume_checkpoint when training."
        )
    try:
        seed = cfg.get("hyper_parameters", {}).get("seed")
        _set_global_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = setup(cfg)
        train_loader, val_loader = _build_train_val_loaders(train_loader, cfg)
        first_batch = next(iter(train_loader))

        ns_meta = getattr(train_loader, "ns_meta", None)
        ns_mode = str((ns_meta or {}).get("ns_mode", "")).lower()
        if ns_mode == "autoregressive":
            h, w = (int(ns_meta["shapelist"][0]), int(ns_meta["shapelist"][1]))
            grid_dtype = first_batch["x"].dtype
        else:
            sample_x = first_batch["x"].squeeze(-1).squeeze(-1)
            while sample_x.dim() > 3:
                sample_x = sample_x[:, 0]
            h, w = int(sample_x.shape[1]), int(sample_x.shape[2])
            grid_dtype = first_batch["x"].dtype

        runtime_overrides = {"shapelist": (h, w)}
        if ns_mode == "autoregressive":
            runtime_overrides.update(
                {
                    "task": "dynamic_autoregressive",
                    "T_in": int(ns_meta["t_in"]),
                    "T_out": int(ns_meta["t_out"]),
                    "out_dim": int(ns_meta["out_dim"]),
                    "fun_dim": int(ns_meta["fun_dim"]),
                }
            )

        checkpoint_path = resume_checkpoint or from_checkpoint
        model, _args = create_model(
            cfg,
            device=device,
            from_checkpoint=checkpoint_path,
            runtime_overrides=runtime_overrides,
        )
        use_aux = bool(getattr(model, "supports_aux", False))

        train_cfg = cfg["hyper_parameters"]
        wandb_cfg = cfg["wandb_logging"]
        max_steps = _normalize_max_steps(train_cfg.get("max_steps"))
        eval_max_steps = _normalize_max_steps(train_cfg.get("eval_max_steps"))
        num_epochs = int(
            _resolve_hparam(train_cfg, _args, "num_epochs", getattr(_args, "epochs", 1))
        )
        learning_rate = float(
            _resolve_hparam(
                train_cfg, _args, "learning_rate", getattr(_args, "lr", 1e-3)
            )
        )
        optimizer_name = str(_resolve_hparam(train_cfg, _args, "optimizer", "adamw"))
        weight_decay = _resolve_hparam(
            train_cfg, _args, "weight_decay", getattr(_args, "weight_decay", None)
        )
        max_grad_norm = _resolve_hparam(
            train_cfg, _args, "max_grad_norm", getattr(_args, "max_grad_norm", None)
        )
        teacher_forcing = bool(
            int(
                _resolve_hparam(
                    train_cfg,
                    _args,
                    "teacher_forcing",
                    getattr(_args, "teacher_forcing", 1),
                )
            )
        )
        scheduler_cfg = {
            "scheduler": _resolve_hparam(
                train_cfg, _args, "scheduler", getattr(_args, "scheduler", "none")
            ),
            "pct_start": float(
                _resolve_hparam(
                    train_cfg, _args, "pct_start", getattr(_args, "pct_start", 0.3)
                )
            ),
            "step_size": int(
                _resolve_hparam(
                    train_cfg, _args, "step_size", getattr(_args, "step_size", 100)
                )
            ),
            "gamma": float(
                _resolve_hparam(train_cfg, _args, "gamma", getattr(_args, "gamma", 0.5))
            ),
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }

        _write_run_config_snapshots(
            cfg,
            _args,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            teacher_forcing=teacher_forcing,
            scheduler_cfg=scheduler_cfg,
        )

        grid = make_grid(h, w, device=device, dtype=grid_dtype)
        grid_flat = grid.view(h * w, -1) if grid.dim() == 3 else grid

        optimizer = build_optimizer(
            model,
            optimizer_name,
            learning_rate,
            weight_decay=weight_decay,
        )
        scheduler, scheduler_step_per_batch = _build_scheduler(
            optimizer, scheduler_cfg, len(train_loader)
        )

        ckpt_prefix = checkpoint_prefix(cfg)
        es_cfg = train_cfg.get("early_stopping", {}) or {}
        es_enabled = bool(es_cfg.get("enabled", False))
        es_patience = int(es_cfg.get("patience", 20))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        es_monitor = str(es_cfg.get("monitor", "val_rel_l2"))
        start_epoch = 0
        best_val_rel_l2 = float("inf")
        best_monitor = float("inf")
        stale_epochs = 0
        if resume_checkpoint is not None:
            resume_state = _restore_training_state(
                resume_checkpoint,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = int(resume_state["start_epoch"])
            best_val_rel_l2 = float(resume_state["best_val_rel_l2"])
            best_monitor = float(resume_state["best_monitor"])
            stale_epochs = int(resume_state["stale_epochs"])

        if start_epoch >= num_epochs:
            print(
                f"Checkpoint already completed {start_epoch} epochs, which is >= "
                f"configured num_epochs={num_epochs}. Nothing to do."
            )
            return best_val_rel_l2

        if ns_mode == "autoregressive":
            t_out = int(ns_meta["t_out"])
            out_dim = int(ns_meta["out_dim"])
            max_val_idx = len(val_loader) - 1
            if eval_max_steps is not None:
                max_val_idx = min(max_val_idx, max(eval_max_steps - 1, 0))

            for epoch in range(start_epoch, num_epochs):
                model.train()
                train_rel_l2_step_sum = 0.0
                train_rel_l2_full_sum = 0.0
                samples_seen = 0

                for step, batch in enumerate(train_loader):
                    coords, fx, target = _prepare_batch_autoregressive(
                        batch, grid_flat, device=device
                    )
                    pred, step_rel_l2, step_stats = _rollout_autoregressive(
                        model,
                        coords,
                        fx,
                        use_aux=use_aux,
                        t_out=t_out,
                        out_dim=out_dim,
                        target=target,
                        teacher_forcing=teacher_forcing,
                    )

                    loss = step_rel_l2
                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(max_grad_norm)
                        )
                    optimizer.step()
                    if scheduler is not None and scheduler_step_per_batch:
                        scheduler.step()

                    batch_size = int(target.shape[0])
                    samples_seen += batch_size
                    train_rel_l2_step_sum += float(step_rel_l2.item())
                    train_rel_l2_full_sum += float(
                        _relative_l2_sum(pred, target).item()
                    )

                    if (step + 1) % wandb_cfg["log_every"] == 0 and wandb_cfg["wandb"]:
                        import wandb

                        global_step = epoch * len(train_loader) + step + 1
                        payload = {
                            "train/step_rel_l2_loss": float(step_rel_l2.item())
                            / max(batch_size * t_out, 1),
                            "train/pred_mean": step_stats["pred_mean"],
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        }
                        if step_stats["pred_base_mean"] is not None:
                            payload["train/pred_base_mean"] = step_stats[
                                "pred_base_mean"
                            ]
                        if step_stats["corr_mean"] is not None:
                            payload["train/corr_mean"] = step_stats["corr_mean"]
                        wandb.log(payload)

                    if max_steps is not None and (step + 1) >= max_steps:
                        break

                if scheduler is not None and not scheduler_step_per_batch:
                    scheduler.step()

                train_epoch_step_rel_l2 = train_rel_l2_step_sum / max(
                    samples_seen * t_out, 1
                )
                train_epoch_full_rel_l2 = train_rel_l2_full_sum / max(samples_seen, 1)
                print(
                    f"Epoch {epoch + 1}: train_step_rel_l2={train_epoch_step_rel_l2:.6f}, "
                    f"train_full_rel_l2={train_epoch_full_rel_l2:.6f}"
                )

                if wandb_cfg["wandb"]:
                    import wandb

                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/epoch_step_rel_l2": train_epoch_step_rel_l2,
                            "train/epoch_full_rel_l2": train_epoch_full_rel_l2,
                        }
                    )

                if wandb_cfg["wandb"]:
                    model.eval()
                    with torch.no_grad():
                        for val_step, batch in enumerate(val_loader):
                            coords, fx, target = _prepare_batch_autoregressive(
                                batch, grid_flat, device=device
                            )
                            pred, _step_rel_l2, step_stats = _rollout_autoregressive(
                                model,
                                coords,
                                fx,
                                use_aux=use_aux,
                                t_out=t_out,
                                out_dim=out_dim,
                                target=target,
                                teacher_forcing=False,
                            )

                            if (
                                epoch % wandb_cfg["image_log_every"] == 0
                                and val_step == random.randint(0, max_val_idx)
                            ):
                                log_prediction_images(
                                    pred[..., :out_dim],
                                    target[..., :out_dim],
                                    fx[..., -out_dim:],
                                    h,
                                    w,
                                    prefix="validation",
                                    epoch=epoch,
                                )

                            if (val_step + 1) % wandb_cfg["log_every"] == 0:
                                import wandb

                                payload = {
                                    "val/pred_mean": step_stats["pred_mean"],
                                    "epoch": epoch + 1,
                                }
                                if step_stats["pred_base_mean"] is not None:
                                    payload["val/pred_base_mean"] = step_stats[
                                        "pred_base_mean"
                                    ]
                                if step_stats["corr_mean"] is not None:
                                    payload["val/corr_mean"] = step_stats["corr_mean"]
                                wandb.log(payload)

                val_metrics = _evaluate_loader_autoregressive(
                    model,
                    val_loader,
                    grid_flat,
                    device,
                    use_aux,
                    t_out,
                    out_dim,
                    max_steps=eval_max_steps,
                )
                val_epoch_loss = float(val_metrics["mse"])
                val_epoch_rel_l2 = float(val_metrics["rel_l2"])
                best_val_rel_l2 = min(best_val_rel_l2, val_epoch_rel_l2)
                monitor_value = _monitor_value(es_monitor, val_metrics)
                if monitor_value < (best_monitor - es_min_delta):
                    best_monitor = monitor_value
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                print(
                    f"Epoch {epoch + 1}: val_mse={val_epoch_loss:.6f}, "
                    f"val_rel_l2={val_epoch_rel_l2:.6f}, "
                    f"val_step_rel_l2={float(val_metrics['step_rel_l2']):.6f}, "
                    f"{es_monitor}={monitor_value:.6f}, "
                    f"best_val_rel_l2={best_val_rel_l2:.6f}"
                )

                if wandb_cfg["wandb"]:
                    import wandb

                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "val/epoch_mse": val_epoch_loss,
                            "val/epoch_rel_l2": val_epoch_rel_l2,
                            "val/epoch_step_rel_l2": float(val_metrics["step_rel_l2"]),
                            "val/best_rel_l2": best_val_rel_l2,
                            "early_stopping/monitor": monitor_value,
                            "early_stopping/best_monitor": best_monitor,
                            "early_stopping/stale_epochs": stale_epochs,
                        }
                    )

                _save_training_checkpoint(
                    f"{cfg['path']['save_dir']}/{ckpt_prefix}_epoch{epoch + 1}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    best_val_rel_l2=best_val_rel_l2,
                    best_monitor=best_monitor,
                    stale_epochs=stale_epochs,
                    metrics={
                        "train_step_rel_l2": train_epoch_step_rel_l2,
                        "train_full_rel_l2": train_epoch_full_rel_l2,
                        "val_mse": val_epoch_loss,
                        "val_rel_l2": val_epoch_rel_l2,
                        "val_step_rel_l2": float(val_metrics["step_rel_l2"]),
                    },
                )

                if es_enabled and stale_epochs >= es_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement in '{es_monitor}' for {es_patience} epochs."
                    )
                    break
        else:
            criterion = nn.MSELoss()
            max_val_idx = len(val_loader) - 1
            if eval_max_steps is not None:
                max_val_idx = min(max_val_idx, max(eval_max_steps - 1, 0))

            for epoch in range(start_epoch, num_epochs):
                model.train()
                total_loss = 0.0
                steps_ran = 0

                for step, batch in enumerate(train_loader):
                    coords, fx, target = prepare_batch(batch, grid, device=device)
                    out = forward_with_optional_aux(model, coords, fx, use_aux=use_aux)
                    pred = out["pred"]

                    loss = criterion(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(max_grad_norm)
                        )
                    optimizer.step()
                    if scheduler is not None and scheduler_step_per_batch:
                        scheduler.step()
                    total_loss += loss.item()
                    steps_ran += 1

                    if (step + 1) % wandb_cfg["log_every"] == 0 and wandb_cfg["wandb"]:
                        import wandb

                        global_step = epoch * len(train_loader) + step + 1
                        payload = {
                            "train/step_loss": loss.item(),
                            "train/pred_mean": out["pred_mean"],
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        }
                        if out["pred_base_mean"] is not None:
                            payload["train/pred_base_mean"] = out["pred_base_mean"]
                        if out["corr_mean"] is not None:
                            payload["train/corr_mean"] = out["corr_mean"]
                        wandb.log(payload)

                    if max_steps is not None and (step + 1) >= max_steps:
                        break

                if scheduler is not None and not scheduler_step_per_batch:
                    scheduler.step()

                avg_loss = total_loss / max(steps_ran, 1)
                print(f"Epoch {epoch + 1}: loss={avg_loss:.6f}")

                if wandb_cfg["wandb"]:
                    import wandb

                    wandb.log({"epoch": epoch + 1, "train/epoch_loss": avg_loss})

                if wandb_cfg["wandb"]:
                    model.eval()
                    with torch.no_grad():
                        for val_step, batch in enumerate(val_loader):
                            coords, fx, target = prepare_batch(
                                batch, grid, device=device
                            )
                            out = forward_with_optional_aux(
                                model, coords, fx, use_aux=use_aux
                            )
                            pred = out["pred"]

                            if (
                                epoch % wandb_cfg["image_log_every"] == 0
                                and val_step == random.randint(0, max_val_idx)
                            ):
                                log_prediction_images(
                                    pred,
                                    target,
                                    fx,
                                    h,
                                    w,
                                    prefix="validation",
                                    epoch=epoch,
                                )

                            if (val_step + 1) % wandb_cfg["log_every"] == 0:
                                import wandb

                                payload = {
                                    "val/pred_mean": out["pred_mean"],
                                    "epoch": epoch + 1,
                                }
                                if out["pred_base_mean"] is not None:
                                    payload["val/pred_base_mean"] = out[
                                        "pred_base_mean"
                                    ]
                                if out["corr_mean"] is not None:
                                    payload["val/corr_mean"] = out["corr_mean"]
                                wandb.log(payload)

                val_metrics = _evaluate_loader(
                    model,
                    val_loader,
                    grid,
                    device,
                    use_aux,
                    criterion,
                    max_steps=eval_max_steps,
                )
                val_epoch_loss = float(val_metrics["mse"])
                val_epoch_rel_l2 = float(val_metrics["rel_l2"])
                best_val_rel_l2 = min(best_val_rel_l2, val_epoch_rel_l2)
                monitor_value = _monitor_value(es_monitor, val_metrics)
                if monitor_value < (best_monitor - es_min_delta):
                    best_monitor = monitor_value
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                print(
                    f"Epoch {epoch + 1}: val_mse={val_epoch_loss:.6f}, "
                    f"val_rel_l2={val_epoch_rel_l2:.6f}, "
                    f"{es_monitor}={monitor_value:.6f}, "
                    f"best_val_rel_l2={best_val_rel_l2:.6f}"
                )

                if wandb_cfg["wandb"]:
                    import wandb

                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "val/epoch_mse": val_epoch_loss,
                            "val/epoch_rel_l2": val_epoch_rel_l2,
                            "val/best_rel_l2": best_val_rel_l2,
                            "early_stopping/monitor": monitor_value,
                            "early_stopping/best_monitor": best_monitor,
                            "early_stopping/stale_epochs": stale_epochs,
                        }
                    )

                _save_training_checkpoint(
                    f"{cfg['path']['save_dir']}/{ckpt_prefix}_epoch{epoch + 1}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    best_val_rel_l2=best_val_rel_l2,
                    best_monitor=best_monitor,
                    stale_epochs=stale_epochs,
                    metrics={
                        "train_loss": avg_loss,
                        "val_mse": val_epoch_loss,
                        "val_rel_l2": val_epoch_rel_l2,
                    },
                )

                if es_enabled and stale_epochs >= es_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement in '{es_monitor}' for {es_patience} epochs."
                    )
                    break

        return best_val_rel_l2
    finally:
        finish_wandb_if_active()


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
    catch_trial_errors = tuple(
        {
            RuntimeError if str(name) == "RuntimeError" else ValueError
            for name in optuna_cfg.get("catch_errors", ["RuntimeError", "ValueError"])
        }
    )

    save_base = optuna_cfg.get("save_dir", cfg["path"]["save_dir"])
    run_name_base = optuna_cfg.get("run_name", "galerkin_optuna")
    wandb_overrides = optuna_cfg.get("wandb_logging", {"wandb": False})

    trial_num = 0

    def _sync_latent_head_dims(trial_cfg: dict) -> dict:
        model_cfg = trial_cfg.setdefault("model", {})
        hc_overrides = model_cfg.get("hc_overrides") or {}
        mode = str(hc_overrides.get("mode", "")).lower()
        if mode != "latent_head":
            return trial_cfg
        n_hidden = model_cfg.get("args", {}).get("n_hidden")
        if n_hidden is None:
            return trial_cfg
        hc_overrides["latent_dim"] = int(n_hidden)
        model_cfg["hc_overrides"] = hc_overrides
        return trial_cfg

    def objective(trial):
        nonlocal trial_num
        trial_num += 1

        trial_cfg = apply_optuna_search_space(cfg, trial, search_space)
        trial_cfg = deepcopy(trial_cfg)
        trial_cfg = _sync_latent_head_dims(trial_cfg)
        trial_cfg["path"]["save_dir"] = f"{save_base}/optuna_{trial_num}"
        trial_cfg["wandb_logging"] = {
            **trial_cfg.get("wandb_logging", {}),
            **wandb_overrides,
            "run_name": f"{run_name_base}_{trial_num}",
        }
        return train(cfg=trial_cfg, from_checkpoint=None)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=num_trials, catch=catch_trial_errors)
    return study
