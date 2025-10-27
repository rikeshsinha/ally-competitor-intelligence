# Eval Suite (Prompt & Rules)

This folder contains a repeatable evaluation setup for the Streamlit app.
It uses a scripted prompt (system+user), small golden CSVs, a JSON schema for model outputs,
and the **active rules** you provided.

## Quick start
```bash
cd eval
pip install promptfoo jsonschema
promptfoo eval -c test_config.promptfoo.yml --output results/
promptfoo view results/
```

> **Note**: The app itself uses a stricter 200-char description cap (Pet Supplies demo).
> The eval rules here come from `active_rules.json` (2000 chars) as provided.

## Files
- `active_rules.json` — rules used by eval (your provided JSON)
- `schema.json` — JSON Schema for the model output (title/bullets/description/rationales)
- `test_config.promptfoo.yml` — prompt + test cases + assertions
- `rubric_quality.md` — LLM-as-a-judge rubric (optional step)
- `golden_csvs/*.csv` — tiny/edge/injection test cases
- `run_eval.sh` — one-liner to run the eval locally

## Results
- After running, check `/eval/results/` for a summary and per-case outputs.
