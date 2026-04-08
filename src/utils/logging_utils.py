import matplotlib.pyplot as plt
import torch

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def init_wandb_if_enabled(cfg: dict):
    wandb_cfg = cfg["wandb_logging"]
    if not wandb_cfg.get("wandb", False):
        return
    if wandb is None:
        print("W&B disabled: wandb package is not available.")
        wandb_cfg["wandb"] = False
        return
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
        wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg["run_name"],
            config=cfg,
            reinit=True,
        )
    except Exception as exc:
        print(f"W&B init failed ({type(exc).__name__}): {exc}")
        print("Continuing with wandb logging disabled.")
        wandb_cfg["wandb"] = False


def finish_wandb_if_active():
    if wandb is None:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()


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
    if wandb is None:
        return
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
