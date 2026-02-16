from train_garlekin import train
import yaml


if __name__ == "__main__":
    with open("config/test_cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f) 
    train(cfg=cfg, from_checkpoint=None)