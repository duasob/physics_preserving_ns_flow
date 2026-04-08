import argparse
from typing import Any, Dict


def build_nsl_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Neural Solver Library Defaults")

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--derivloss", type=bool, default=False)
    parser.add_argument("--teacher_forcing", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="OneCycleLR")
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)

    # data
    parser.add_argument("--data_path", type=str, default="/data/fno/")
    parser.add_argument("--loader", type=str, default="airfoil")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--norm_type", type=str, default="UnitTransformer")
    parser.add_argument("--geotype", type=str, default="unstructured")
    parser.add_argument("--time_input", type=bool, default=False)
    parser.add_argument("--space_dim", type=int, default=2)
    parser.add_argument("--fun_dim", type=int, default=0)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--shapelist", type=list, default=None)
    parser.add_argument("--downsamplex", type=int, default=1)
    parser.add_argument("--downsampley", type=int, default=1)
    parser.add_argument("--downsamplez", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.2)

    # task
    parser.add_argument("--task", type=str, default="steady")
    parser.add_argument("--T_in", type=int, default=10)
    parser.add_argument("--T_out", type=int, default=10)

    # model
    parser.add_argument("--model", type=str, default="Transolver")
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--act", type=str, default="gelu")
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--unified_pos", type=int, default=0)
    parser.add_argument("--ref", type=int, default=8)

    # model specific
    parser.add_argument("--slice_num", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--psi_dim", type=int, default=8)
    parser.add_argument("--attn_type", type=str, default="nystrom")
    parser.add_argument("--mwt_k", type=int, default=3)

    # eval
    parser.add_argument("--eval", type=int, default=0)
    parser.add_argument("--save_name", type=str, default="Transolver_check")
    parser.add_argument("--vis_num", type=int, default=10)
    parser.add_argument("--vis_bound", type=int, nargs="+", default=None)

    return parser


def get_nsl_default_args() -> Dict[str, Any]:
    parser = build_nsl_default_parser()
    return vars(parser.parse_args([]))
