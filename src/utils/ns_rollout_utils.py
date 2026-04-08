from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch

from src.utils.train_utils import forward_with_optional_aux


def save_rollout_animation(target, pred, error, save_path="images/ns_rollout.gif", fps=5):
    target = np.asarray(target)
    pred = np.asarray(pred)
    error = np.asarray(error)
    if target.ndim != 3 or pred.ndim != 3 or error.ndim != 3:
        raise ValueError(
            "Expected target, pred, error to have shape (T, H, W). "
            f"Got {target.shape}, {pred.shape}, {error.shape}."
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = matplotlib.colormaps["jet"]
    fig, ax = plt.subplots(1, 3, figsize=[9, 3])

    ax[0].set_title("Target")
    im0 = ax[0].imshow(target[0], cmap=cmap)
    ax[1].set_title("Prediction")
    im1 = ax[1].imshow(pred[0], cmap=cmap)
    ax[2].set_title("Abs Error")
    im2 = ax[2].imshow(error[0], cmap=cmap)
    title = fig.suptitle("t=0")
    fig.tight_layout()
    fig.colorbar(im1, ax=ax)

    frames = min(len(target), len(pred), len(error))

    def animate(frame_idx):
        y = target[frame_idx]
        p = pred[frame_idx]
        e = error[frame_idx]

        im0.set_data(y)
        im1.set_data(p)
        im2.set_data(e)

        vmin = min(float(np.min(y)), float(np.min(p)))
        vmax = max(float(np.max(y)), float(np.max(p)))
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        im2.set_clim(0.0, max(float(np.max(e)), 1e-12))
        title.set_text(f"t={frame_idx}")

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=max(int(1000 / max(int(fps), 1)), 1),
        blit=False,
        frames=frames,
        repeat_delay=1000,
    )
    if save_path.suffix.lower() == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    ani.save(str(save_path), writer=writer)
    plt.close(fig)


def _prepare_grid(grid, h, w, device):
    if grid.dim() == 3:
        grid_flat = grid.view(h * w, -1)
    else:
        grid_flat = grid
    return grid_flat.to(device)


def _resolve_hw(grid, shapelist=None):
    if shapelist is not None:
        return int(shapelist[0]), int(shapelist[1])
    if grid.dim() == 3:
        return int(grid.shape[0]), int(grid.shape[1])
    raise ValueError(
        "shapelist is required when grid is already flattened to (H*W, 2)."
    )


@torch.no_grad()
def rollout_predict_ns_torch(
    model,
    batch,
    grid,
    *,
    steps=None,
    device="cuda",
    shapelist=None,
    out_dim=1,
    use_aux=False,
    sample_idx=0,
    channel_idx=0,
):
    model.eval()
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(
            "NS rollout expects x and y with shape (B, N, C). "
            f"Got {tuple(x.shape)} and {tuple(y.shape)}."
        )

    h, w = _resolve_hw(grid, shapelist=shapelist)
    b, n, history_channels = x.shape
    if n != h * w:
        raise ValueError(
            f"Grid shape {(h, w)} implies {h * w} points, but batch has {n}."
        )
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")
    if history_channels < out_dim:
        raise ValueError(
            f"Input history channel dim ({history_channels}) must be >= out_dim ({out_dim})."
        )
    if sample_idx < 0 or sample_idx >= b:
        raise ValueError(f"sample_idx={sample_idx} is invalid for batch size {b}.")
    if channel_idx < 0 or channel_idx >= out_dim:
        raise ValueError(
            f"channel_idx={channel_idx} is invalid for out_dim={out_dim}."
        )

    total_steps = y.shape[-1] // out_dim
    if total_steps <= 0:
        raise ValueError(
            f"Target channel dim ({y.shape[-1]}) is incompatible with out_dim={out_dim}."
        )
    steps = total_steps if steps is None else min(int(steps), total_steps)

    grid_flat = _prepare_grid(grid, h, w, device)
    coords = grid_flat.unsqueeze(0).repeat(b, 1, 1)

    current_fx = x
    preds = []
    targets = []
    for t in range(steps):
        out = forward_with_optional_aux(model, coords, current_fx, use_aux=use_aux)
        pred_t = out["pred"]
        tgt_t = y[..., out_dim * t : out_dim * (t + 1)]
        preds.append(
            pred_t.view(b, h, w, out_dim)[sample_idx, :, :, channel_idx].detach().cpu()
        )
        targets.append(
            tgt_t.view(b, h, w, out_dim)[sample_idx, :, :, channel_idx].detach().cpu()
        )
        current_fx = torch.cat((current_fx[..., out_dim:], pred_t), dim=-1)

    pred = torch.stack(preds, dim=0).numpy()
    target = torch.stack(targets, dim=0).numpy()
    return pred, target


def visual_rollout_ns_torch(
    model,
    batch,
    grid,
    *,
    steps=None,
    device="cuda",
    shapelist=None,
    out_dim=1,
    use_aux=False,
    sample_idx=0,
    channel_idx=0,
    save_path="images/ns_rollout.gif",
    fps=5,
):
    pred, target = rollout_predict_ns_torch(
        model,
        batch,
        grid,
        steps=steps,
        device=device,
        shapelist=shapelist,
        out_dim=out_dim,
        use_aux=use_aux,
        sample_idx=sample_idx,
        channel_idx=channel_idx,
    )
    error = np.abs(pred - target)
    save_rollout_animation(target, pred, error, save_path=save_path, fps=fps)
    return pred, target
