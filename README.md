# Catalyst Agent

Standalone adsorption-site optimization agent extracted from the original research workspace.

## What is included

- `main_adsorb_agent.py` as the main entry point
- `adsorb_agent_core/` agent workflow, LLM chains, analysis, and engine code
- `config/` YAML experiment configs
- `reasoning/` prompt/reasoning text
- `checkpoints/` local GNN checkpoints from the source workspace
- `fairchem-forked/src/` local FairChem source dependency

Large experiment outputs, caches, personal scratch files, and API keys are intentionally excluded.
FairChem's license is kept at `fairchem-forked/LICENSE.md`.

## Setup

```bash
cd catalyst-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` or export the needed environment variables in your shell.

```bash
export CLAUDE_API_KEY="..."
export QWEN_API_KEY="..."
```

## Data

The default configs use relative paths. Add your metadata and FairChem database files here:

- `data/processed/updated_sid_to_details.pkl`
- `fairchem-forked/src/fairchem/data/oc/databases/pkls/bulks.pkl`
- `fairchem-forked/src/fairchem/data/oc/databases/pkls/adsorbates.pkl`

## Run

```bash
python main_adsorb_agent.py --path config/adsorb_agent_demo_claude_rep3.yaml
```

Results are written under `results/`, which is ignored by Git.
