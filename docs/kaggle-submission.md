# Sol Kaggle PTCG submission and common-engine verification

Sol has two explicit runtime boundaries:

- `ptcg-agent run` keeps the development/training JSONL contract (`legal_actions` in, one action out).
- `main.agent(obs_dict) -> list[int]` is the Kaggle competition entry point. It returns `deck.csv` on
  the initial call and legal option indices for subsequent selection prompts.

The 60-card deck is the strongest shared measured baseline currently used by the matsu/fable line.
It is retained verbatim so Sol policy behavior can be compared without confounding the experiment by
changing the deck. Card counts and engine legality are checked by the real-engine startup below.

Competition runtime files are license-restricted and therefore never committed. Install the same
official runtime used by the other agents, build the archive, and run a full match as follows:

```bash
SRC=/workspaces/kaggle-ptcg-matsu/data/simulation/extracted bash scripts/setup_engine.sh
bash scripts/build_submission.sh
python eval/run_common_match.py \
  --opponent /workspaces/ptcg-agent-fable \
  --log artifacts/sol-vs-fable.jsonl
```

Success prints a JSON object with `status: completed`; the JSONL log records every selected action and
the terminal winner/decision count. `cg/`, `data/`, `submission.tar.gz`, and `artifacts/` remain local
or ignored so competition assets and generated logs are not redistributed.
