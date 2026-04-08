from copy import deepcopy


def _set_by_dotted_key(root, dotted_key, value):
    keys = str(dotted_key).split(".")
    node = root
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def suggest_value(trial, name, spec):
    if isinstance(spec, dict):
        spec_type = spec.get("type", "").lower()
        if spec_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        if spec_type == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if spec_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        raise ValueError(f"Unknown optuna spec type for {name}: {spec_type}")

    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    raise ValueError(f"Invalid optuna spec for {name}: {spec}")


def apply_optuna_search_space(cfg, trial, search_space):
    trial_cfg = deepcopy(cfg)
    for name, spec in search_space.items():
        value = suggest_value(trial, name, spec)
        if "." in str(name):
            _set_by_dotted_key(trial_cfg, name, value)
        else:
            hp = trial_cfg.setdefault("hyper_parameters", {})
            hp[name] = value
    return trial_cfg
