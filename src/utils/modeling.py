import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import yaml

from src.utils.nsl_parser_defaults import get_nsl_default_args

MODEL_REQUIRED_ARGS = {
    "Galerkin_Transformer": [
        "n_hidden",
        "n_heads",
        "dropout",
        "mlp_ratio",
        "n_layers",
        "out_dim",
    ],
    "FNO": [
        "n_hidden",
        "modes",
        "out_dim",
    ],
}


def ensure_nsl_path() -> None:
    repo_root = Path("src/Neural-Solver-Library").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _resolve_backbone(cfg: dict) -> str:
    backbone = cfg.get("model", {}).get("backbone", "Galerkin_Transformer")
    if isinstance(backbone, dict):
        backbone = next(iter(backbone.keys()), "Galerkin_Transformer")
    return str(backbone)


def _resolve_model_config_path(cfg: dict, backbone: str) -> Optional[Path]:
    model_cfg = cfg.get("model", {})
    explicit_path = model_cfg.get("config")
    if explicit_path:
        return Path(str(explicit_path))
    default_path = Path("config/model_config") / f"{backbone}.yaml"
    if default_path.exists():
        return default_path
    return None


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in YAML config: {path}")
    return data


def load_model_config(cfg: dict, backbone: str) -> Dict[str, Any]:
    model_config_path = _resolve_model_config_path(cfg, backbone)
    if model_config_path is None:
        return {}

    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {model_config_path}. "
            f"Set model.config explicitly or add a default mapping for {backbone}."
        )
    return _load_yaml_dict(model_config_path)


def load_hc_config(cfg: dict) -> Optional[dict]:
    hc_spec = cfg.get("model", {}).get("hc")
    if not hc_spec:
        return None

    hc_text = str(hc_spec)
    if hc_text.endswith((".yaml", ".yml")):
        cfg_path = Path(hc_text)
    else:
        cfg_path = Path("config/hc_config") / f"{hc_text}_cfg.yaml"
    hc_cfg = _load_yaml_dict(cfg_path)
    hc_overrides = cfg.get("model", {}).get("hc_overrides", {})
    if hc_overrides:
        hc_cfg.update(hc_overrides)
    return hc_cfg


def _resolve_hc_kind(cfg: dict) -> Optional[str]:
    hc_spec = cfg.get("model", {}).get("hc")
    if not hc_spec:
        return None
    hc_text = str(hc_spec)
    if hc_text.endswith((".yaml", ".yml")):
        stem = Path(hc_text).stem
        if stem.endswith("_cfg"):
            stem = stem[: -len("_cfg")]
        return stem
    return hc_text


def _validate_required_args(backbone: str, args_dict: Dict[str, Any]) -> None:
    required = MODEL_REQUIRED_ARGS.get(backbone, [])
    missing = [name for name in required if args_dict.get(name) is None]
    if missing:
        raise ValueError(
            f"Missing required args for {backbone}: {missing}. "
            "Add them in model config YAML or model.args overrides."
        )


def build_model_args(cfg: dict, runtime_overrides: Optional[Dict[str, Any]] = None):
    backbone = _resolve_backbone(cfg)
    args_dict = get_nsl_default_args()
    args_dict["model"] = backbone

    file_model_cfg = load_model_config(cfg, backbone)
    args_dict.update(file_model_cfg)

    # Optional inline overrides in the main train config.
    args_dict.update(cfg.get("model", {}).get("args", {}))
    if runtime_overrides:
        args_dict.update(runtime_overrides)

    shapelist = args_dict.get("shapelist")
    if isinstance(shapelist, list):
        args_dict["shapelist"] = tuple(shapelist)

    _validate_required_args(backbone, args_dict)
    return SimpleNamespace(**args_dict)


def create_model(
    cfg: dict,
    device,
    from_checkpoint: Optional[str] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    checkpoint_load_mode: str = "strict",
) -> Tuple[object, object]:
    ensure_nsl_path()
    import torch
    from models.model_factory import get_model

    from src.hard_constraints import build_mean_constraint_wrapper

    args = build_model_args(cfg, runtime_overrides=runtime_overrides)
    model = get_model(args).to(device)

    hc_kind = _resolve_hc_kind(cfg)
    hc_cfg = load_hc_config(cfg) if hc_kind else None
    if hc_kind:
        if hc_kind != "mean_correction":
            raise ValueError(
                f"Unsupported hard constraint '{hc_kind}'. "
                "Currently supported: mean_correction"
            )
        model = build_mean_constraint_wrapper(model, args, hc_cfg or {}).to(device)

    if from_checkpoint is not None:
        checkpoint_state = torch.load(from_checkpoint, map_location=device)
        if not (
            isinstance(checkpoint_state, dict)
            and isinstance(checkpoint_state.get("model_state_dict"), dict)
        ):
            raise RuntimeError(
                "Checkpoint load failed. Expected the new resumable checkpoint "
                "format with a top-level 'model_state_dict'. Regenerate the "
                "checkpoint with the current training scripts."
            )
        checkpoint_state = checkpoint_state["model_state_dict"]

        try:
            model.load_state_dict(checkpoint_state)
            print(
                f"Loaded model from checkpoint: {from_checkpoint} "
                f"(mode={checkpoint_load_mode})"
            )
        except RuntimeError as exc:
            mode = str(checkpoint_load_mode).strip().lower()
            if mode != "backbone":
                raise RuntimeError(
                    "Checkpoint load failed. This usually means the config used to "
                    "rebuild the model does not match the checkpoint architecture. "
                    "Use the exact training config, or explicitly request "
                    "checkpoint_load_mode='backbone' if you only want to reuse the "
                    "backbone weights."
                ) from exc

            if not hasattr(model, "backbone"):
                raise RuntimeError(
                    "checkpoint_load_mode='backbone' was requested, but the model "
                    "does not expose a backbone module."
                ) from exc

            backbone_state = {
                key[len("backbone.") :]: value
                for key, value in checkpoint_state.items()
                if isinstance(key, str) and key.startswith("backbone.")
            }
            if not backbone_state:
                raise RuntimeError(
                    "Backbone-only checkpoint loading was requested, but the "
                    "checkpoint does not contain any 'backbone.'-prefixed keys."
                ) from exc

            incompatible = model.backbone.load_state_dict(backbone_state, strict=False)
            print(
                f"Loaded backbone-only weights from checkpoint: {from_checkpoint} "
                f"(missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)})"
            )

    return model, args


def checkpoint_prefix(cfg: dict) -> str:
    backbone = _resolve_backbone(cfg)
    return backbone.lower().replace(" ", "_")
