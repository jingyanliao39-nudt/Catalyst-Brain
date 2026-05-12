import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FAIRCHEM_SRC = PROJECT_ROOT / "fairchem-forked" / "src"


def project_path(path_value):
    """Resolve relative project paths against the repository root."""
    if path_value is None:
        return None
    path_str = os.path.expandvars(os.path.expanduser(str(path_value)))
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def ensure_fairchem_on_path():
    fairchem_src = str(FAIRCHEM_SRC)
    if fairchem_src not in sys.path:
        sys.path.insert(0, fairchem_src)


def resolve_config_paths(config):
    """Resolve path-like config values after loading YAML."""
    paths = config.get("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            paths[key] = project_path(value)

    agent_settings = config.get("agent_settings", {})
    gnn_model = agent_settings.get("gnn_model")
    if isinstance(gnn_model, str) and _looks_like_path(gnn_model):
        agent_settings["gnn_model"] = project_path(gnn_model)

    return config


def env_key(name, default=""):
    return os.environ.get(name, default)


def _looks_like_path(value):
    return (
        value.startswith(".")
        or value.startswith("/")
        or "/" in value
        or "\\" in value
        or value.endswith((".pt", ".pth", ".ckpt"))
    )
