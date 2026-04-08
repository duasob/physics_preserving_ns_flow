import importlib


def get_model(args):
    model_module_map = {
        "PointNet": "models.PointNet",
        "Graph_UNet": "models.Graph_UNet",
        "GraphSAGE": "models.GraphSAGE",
        "MWT": "models.MWT",
        "ONO": "models.ONO",
        "F_FNO": "models.F_FNO",
        "U_FNO": "models.U_FNO",
        "U_NO": "models.U_NO",
        "GNOT": "models.GNOT",
        "Galerkin_Transformer": "models.Galerkin_Transformer",
        "Swin_Transformer": "models.Swin_Transformer",
        "Factformer": "models.Factformer",
        "Transformer": "models.Transformer",
        "Transformer_Spatial_Bias": "models.Transformer_Spatial_Bias",
        "U_Net": "models.U_Net",
        "FNO": "models.FNO",
        "Transolver": "models.Transolver",
        "LSM": "models.LSM",
    }

    if args.model not in model_module_map:
        raise ValueError(
            f"Unsupported model '{args.model}'. Available: {sorted(model_module_map)}"
        )

    model_module = importlib.import_module(model_module_map[args.model])
    return model_module.Model(args)
