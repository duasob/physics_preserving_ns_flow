from typing import Optional, Sequence

import torch.nn as nn

from .mean_correction import build_mlp, match_mean


def _resolve_reduce_dims(tensor, channel_dim, reduce_dims):
    if reduce_dims is not None:
        return tuple(int(d) for d in reduce_dims)
    if channel_dim < 0:
        channel_dim = tensor.ndim + channel_dim
    return tuple(i for i in range(1, tensor.ndim) if i != channel_dim)


def _resolve_module_by_path(module: nn.Module, module_path: str) -> nn.Module:
    current = module
    for part in module_path.split("."):
        if part.lstrip("-").isdigit():
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            raise ValueError(
                f"Invalid latent_module path '{module_path}', missing '{part}'"
            )
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise ValueError(f"Resolved object at '{module_path}' is not an nn.Module")
    return current


class ForwardHookLatentExtractor:
    """
    Utility class to extract latent features from an intermediate module using a forward hook.
    Registers a PyTorch forward hook (register_forward_hook) on the specified module path within the model.
    During the forward pass, it captures the output of that module and stores it as the "latent" features.
    The extracted features can then be used by a constraint head for correction.
    """

    def __init__(self, model: nn.Module, module_path: str):
        self.module_path = module_path
        self.latent = None
        target_module = _resolve_module_by_path(model, module_path)
        self.handle = target_module.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.latent = output[0] if isinstance(output, tuple) else output

    def reset(self):
        self.latent = None

    def get(self):
        return self.latent

    def remove(self):
        self.handle.remove()


class MeanConstraint(nn.Module):
    """
    Constraint wrapper supporting three modes:
    - post_output: strict projection out = pred - mean(pred)
    - post_output_learned: learned correction from pred
    - latent_head: learned correction from latent features
    """

    def __init__(
        self,
        *,
        mode: str,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 0,
        act: str = "gelu",
        latent_dim: Optional[int] = None,
        channel_dim: int = -1,
        reduce_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.mode = mode
        self.channel_dim = channel_dim
        self.reduce_dims = tuple(reduce_dims) if reduce_dims is not None else None

        if mode == "post_output":
            self.corr_head = None
        elif mode == "post_output_learned":
            head_hidden = hidden_dim if hidden_dim is not None else max(2 * out_dim, 8)
            self.corr_head = build_mlp(
                out_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        elif mode == "latent_head":
            if latent_dim is None:
                raise ValueError("latent_dim is required for mode='latent_head'")
            head_hidden = hidden_dim if hidden_dim is not None else latent_dim
            self.corr_head = build_mlp(
                latent_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        else:
            raise ValueError(
                f"Unknown mean constraint mode '{mode}'. "
                "Use one of: post_output, post_output_learned, latent_head."
            )

    def forward(self, *, pred, latent=None, return_aux=False):
        reduce_dims = _resolve_reduce_dims(pred, self.channel_dim, self.reduce_dims)

        if self.mode == "post_output":
            corr = pred.mean(dim=reduce_dims, keepdim=True)
            out = pred - corr
        else:
            if self.mode == "post_output_learned":
                corr_raw = self.corr_head(pred)
            else:
                if latent is None:
                    raise ValueError("latent is required for mode='latent_head'")
                corr_raw = self.corr_head(latent)
            corr = match_mean(
                corr_raw,
                pred,
                channel_dim=self.channel_dim,
                reduce_dims=reduce_dims,
            )
            out = pred - corr

        if return_aux:
            return out, pred, corr
        return out


class ConstrainedModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        constraint: Optional[nn.Module] = None,
        latent_extractor: Optional[ForwardHookLatentExtractor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.constraint = constraint
        self.latent_extractor = latent_extractor
        self.supports_aux = constraint is not None

    def forward(self, *args, return_aux=False, **kwargs):
        if self.constraint is None:
            return self.backbone(*args, **kwargs)

        if self.latent_extractor is not None:
            self.latent_extractor.reset()

        pred = self.backbone(*args, **kwargs)
        latent = None if self.latent_extractor is None else self.latent_extractor.get()
        return self.constraint(pred=pred, latent=latent, return_aux=return_aux)


def _default_latent_module(backbone_name: str) -> Optional[str]:
    defaults = {
        "Galerkin_Transformer": "blocks.-1.ln_3",
    }
    return defaults.get(backbone_name)


def build_mean_constraint_wrapper(
    backbone: nn.Module, args, hc_cfg: dict
) -> ConstrainedModel:
    mode = str(hc_cfg.get("mode", "post_output")).lower()
    channel_dim = int(hc_cfg.get("channel_dim", -1))
    reduce_dims = hc_cfg.get("reduce_dims")
    hidden_dim = hc_cfg.get("correction_hidden")
    n_layers = int(hc_cfg.get("correction_layers", 0))
    act = hc_cfg.get("correction_act", "gelu")

    latent_extractor = None
    latent_dim = hc_cfg.get("latent_dim")
    if mode == "latent_head":
        latent_module = hc_cfg.get("latent_module", _default_latent_module(args.model))
        if latent_module is None:
            raise ValueError(
                "mode='latent_head' requires hc_cfg.latent_module "
                "or a known default for this backbone"
            )
        latent_extractor = ForwardHookLatentExtractor(backbone, latent_module)
        if latent_dim is None:
            latent_dim = getattr(args, "n_hidden", None)
        if latent_dim is None:
            raise ValueError(
                "mode='latent_head' requires hc_cfg.latent_dim or args.n_hidden"
            )

    constraint = MeanConstraint(
        mode=mode,
        out_dim=int(args.out_dim),
        hidden_dim=None if hidden_dim is None else int(hidden_dim),
        n_layers=n_layers,
        act=act,
        latent_dim=None if latent_dim is None else int(latent_dim),
        channel_dim=channel_dim,
        reduce_dims=reduce_dims,
    )

    wrapped = ConstrainedModel(
        backbone=backbone,
        constraint=constraint,
        latent_extractor=latent_extractor,
    )

    if bool(hc_cfg.get("freeze_base", False)):
        for p in wrapped.backbone.parameters():
            p.requires_grad = False

    return wrapped
