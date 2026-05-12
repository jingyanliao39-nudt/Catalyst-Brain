import argparse
from project_paths import ensure_fairchem_on_path

ensure_fairchem_on_path()

from utils import load_config
from adsorb_agent_core.core_06_workflow import multirun_adsorb_aigent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file path')
    parser.add_argument('--path', type=str, metavar='CONFIG_FILE', 
                        help='Path to configuration file', default='config/adsorb_agent_demo_claude_rep3.yaml')
    args = parser.parse_args()

    print(f"[debug] running: {__file__}", flush=True)
    config = load_config(args.path)
    multirun_adsorb_aigent(config)
