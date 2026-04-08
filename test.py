import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from train import (
    _create_nsl_ns_mat_test_loader,
    _evaluate_loader,
    _evaluate_loader_autoregressive,
    _normalize_max_steps,
    _set_global_seed,
)
from src.utils.data_utils import make_grid
from src.utils.modeling import create_model
from src.utils.ns_rollout_utils import rollout_predict_ns_torch, visual_rollout_ns_torch


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_cfg.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--checkpoint-load-mode",
        type=str,
        default="strict",
        choices=["strict", "backbone"],
        help=(
            "How to load the checkpoint. "
            "'strict' requires the model config to exactly match the checkpoint. "
            "'backbone' only reuses backbone.* weights."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | cpu",
    )
    parser.add_argument(
        "--test-root",
        type=str,
        default=None,
        help=(
            "Optional override for the evaluation dataset root. "
            "For NS .mat evaluation this can be a .mat file or a directory containing it."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for test-time batch size.",
    )
    parser.add_argument(
        "--max-steps",
        type=str,
        default=None,
        help="Optional cap on the number of test batches to evaluate.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional output path for a rollout animation (.gif or .mp4).",
    )
    parser.add_argument(
        "--video-batch-index",
        type=int,
        default=0,
        help="Zero-based index of the test batch to visualize.",
    )
    parser.add_argument(
        "--video-sample-idx",
        type=int,
        default=0,
        help="Zero-based sample index within the selected batch.",
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=None,
        help="Optional rollout length for the saved animation.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=5,
        help="Frames per second for the saved animation.",
    )
    parser.add_argument(
        "--video-channel-idx",
        type=int,
        default=0,
        help="Channel index to visualize when out_dim > 1.",
    )
    parser.add_argument(
        "--print-batch-stats",
        action="store_true",
        help="Print prediction/target summary stats for the selected video batch.",
    )
    return parser.parse_args()


def _load_cfg(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict config in {config_path}")
    return cfg


def _resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_test_loader(
    cfg: dict,
    *,
    test_root: str | None = None,
    batch_size: int | None = None,
):
    mat_loader = _create_nsl_ns_mat_test_loader(
        cfg,
        root_dir=test_root,
        batch_size=batch_size,
    )
    if mat_loader is None:
        raise ValueError(
            "This repository now supports only the Navier-Stokes .mat dataset. "
            "Set path.root_dir or --test-root to a .mat file or a directory "
            "containing 'NavierStokes_V1e-5_N1200_T20.mat'."
        )
    return mat_loader


def _build_runtime_context(
    cfg: dict,
    test_loader,
    device,
    checkpoint_path: str,
    checkpoint_load_mode: str,
):
    try:
        first_batch = next(iter(test_loader))
    except StopIteration as exc:
        raise ValueError("Test loader is empty.") from exc

    ns_meta = getattr(test_loader, "ns_meta", None)
    ns_mode = str((ns_meta or {}).get("ns_mode", "")).lower()
    if ns_mode == "autoregressive":
        h, w = int(ns_meta["shapelist"][0]), int(ns_meta["shapelist"][1])
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

    model, model_args = create_model(
        cfg,
        device=device,
        from_checkpoint=checkpoint_path,
        runtime_overrides=runtime_overrides,
        checkpoint_load_mode=checkpoint_load_mode,
    )
    model.eval()

    grid = make_grid(h, w, device=device, dtype=grid_dtype)
    grid_flat = grid.view(h * w, -1) if grid.dim() == 3 else grid
    return {
        "model": model,
        "model_args": model_args,
        "use_aux": bool(getattr(model, "supports_aux", False)),
        "grid": grid,
        "grid_flat": grid_flat,
        "h": h,
        "w": w,
        "ns_meta": ns_meta,
        "ns_mode": ns_mode,
    }


def _evaluate_test_loader(context: dict, test_loader, device, max_steps):
    model = context["model"]
    use_aux = context["use_aux"]
    ns_mode = context["ns_mode"]
    if ns_mode == "autoregressive":
        ns_meta = context["ns_meta"]
        return _evaluate_loader_autoregressive(
            model,
            test_loader,
            context["grid_flat"],
            device,
            use_aux,
            int(ns_meta["t_out"]),
            int(ns_meta["out_dim"]),
            max_steps=max_steps,
        )

    criterion = nn.MSELoss()
    return _evaluate_loader(
        model,
        test_loader,
        context["grid"],
        device,
        use_aux,
        criterion,
        max_steps=max_steps,
    )


def _get_batch_by_index(loader, batch_index: int):
    if batch_index < 0:
        raise ValueError(f"video batch index must be >= 0, got {batch_index}")
    for idx, batch in enumerate(loader):
        if idx == batch_index:
            return batch
    raise IndexError(
        f"Requested video batch index {batch_index}, but loader has fewer batches."
    )


def _save_rollout_video(context: dict, test_loader, args, device):
    batch = _get_batch_by_index(test_loader, int(args.video_batch_index))
    ns_meta = context["ns_meta"] or {}
    pred, target = visual_rollout_ns_torch(
        context["model"],
        batch,
        context["grid_flat"],
        steps=args.video_steps,
        device=device,
        shapelist=(context["h"], context["w"]),
        out_dim=int(ns_meta.get("out_dim", 1)),
        use_aux=context["use_aux"],
        sample_idx=int(args.video_sample_idx),
        channel_idx=int(args.video_channel_idx),
        save_path=args.video_path,
        fps=int(args.video_fps),
    )
    print(
        "Saved rollout video:",
        args.video_path,
        f"frames={pred.shape[0]}",
        f"sample_idx={args.video_sample_idx}",
        f"channel_idx={args.video_channel_idx}",
    )
    return pred, target


def _print_array_stats(name: str, value):
    print(
        f"{name}: "
        f"shape={tuple(value.shape)} "
        f"min={float(value.min()):.6f} "
        f"max={float(value.max()):.6f} "
        f"mean={float(value.mean()):.6f} "
        f"std={float(value.std()):.6f}"
    )


def _print_rollout_stats(context: dict, test_loader, args, device):
    batch = _get_batch_by_index(test_loader, int(args.video_batch_index))
    ns_meta = context["ns_meta"] or {}
    pred, target = rollout_predict_ns_torch(
        context["model"],
        batch,
        context["grid_flat"],
        steps=args.video_steps,
        device=device,
        shapelist=(context["h"], context["w"]),
        out_dim=int(ns_meta.get("out_dim", 1)),
        use_aux=context["use_aux"],
        sample_idx=int(args.video_sample_idx),
        channel_idx=int(args.video_channel_idx),
    )
    _print_array_stats("target", target)
    _print_array_stats("prediction", pred)
    _print_array_stats("abs_error", abs(pred - target))


def main():
    args = _parse_args()
    cfg = deepcopy(_load_cfg(args.config))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _set_global_seed(cfg.get("hyper_parameters", {}).get("seed"))
    device = _resolve_device(args.device)
    test_loader = _build_test_loader(
        cfg,
        test_root=args.test_root,
        batch_size=args.batch_size,
    )
    context = _build_runtime_context(
        cfg,
        test_loader,
        device,
        str(checkpoint_path),
        args.checkpoint_load_mode,
    )

    max_steps = (
        _normalize_max_steps(args.max_steps)
        if args.max_steps is not None
        else _normalize_max_steps(cfg.get("hyper_parameters", {}).get("eval_max_steps"))
    )
    metrics = _evaluate_test_loader(context, test_loader, device, max_steps=max_steps)

    print("Test metrics:")
    print(f"  mse: {float(metrics['mse']):.6f}")
    print(f"  rel_l2: {float(metrics['rel_l2']):.6f}")
    if "step_rel_l2" in metrics:
        print(f"  step_rel_l2: {float(metrics['step_rel_l2']):.6f}")

    if args.video_path:
        _save_rollout_video(context, test_loader, args, device)
    if args.print_batch_stats:
        _print_rollout_stats(context, test_loader, args, device)


if __name__ == "__main__":
    main()
