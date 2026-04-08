import argparse

import yaml

from train import optuna_search, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_cfg.yaml")
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Initialize model weights from a checkpoint and start a fresh training run.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training state from a checkpoint, including optimizer and scheduler.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "optuna"],
        help="Run standard training or Optuna search",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.from_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError(
            "Use only one of --from-checkpoint or --resume-checkpoint."
        )
    if args.mode == "optuna":
        study = optuna_search(cfg=cfg)
        print("Best trial value:", study.best_value)
        print("Best trial params:", study.best_trial.params)
    else:
        train(
            cfg=cfg,
            from_checkpoint=args.from_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
        )
