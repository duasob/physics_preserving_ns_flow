Training writes resumable checkpoints. Each `.pth` contains:
- `model_state_dict` # model inside here
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `best_val_rel_l2`
- `best_monitor`
- `stale_epochs`
- `metrics`