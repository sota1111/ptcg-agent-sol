# Sol semantic-MCTS submission and promotion workflow

Sol has two explicit runtime boundaries:

- `ptcg-agent run` keeps the development/training JSONL contract (`legal_actions` in, one action out).
- `main.agent(obs_dict) -> list[int]` is the Kaggle competition entry point. It returns `deck.csv` on
  the initial call and legal option indices for subsequent selection prompts.

The 60-card deck is the strongest shared measured baseline currently used by the matsu/fable line.
It is retained verbatim so Sol policy behavior can be compared without confounding the experiment by
changing the deck. Card counts and engine legality are checked by the real-engine startup below.

## Policy

`main.agent` uses the board, cards, prompt context, and normalized legal options through four layers:
determinized MCTS, one-ply greedy scoring, an explicit rule table covering every shipped selection
context, and a final legal-action fallback. The search budget steps down after 300 and 420 seconds of
cumulative think time and hands off to greedy at 510 seconds, leaving safety margin inside the
600-second match allowance. `AGENT_SEED` fixes all policy-side random streams.

The pre-change hash policy is frozen under `eval/hash_baseline/` and is never packaged. Both sides use
the same unchanged `deck.csv`, so the first promotion comparison measures policy only.

Competition runtime files are license-restricted and therefore never committed. Install the same
official runtime used by the other agents, build the archive, and run a full match as follows:

```bash
SRC=/workspaces/kaggle-ptcg-matsu/data/simulation/extracted bash scripts/setup_engine.sh
bash scripts/build_submission.sh
python eval/battle_vs.py --seeds 20 --base-seed 20260722 \
  --json artifacts/semantic-vs-hash.json
```

Each seed runs twice with seats reversed. Promotion requires at least 20 seeds, semantic win rate at
least 60% excluding draws, Wilson 95% lower bound strictly above 50%, zero semantic faults, zero
unfinished matches, and maximum semantic think time below 600 seconds. Only when the JSON report says
`"promotion": {"promote": true}` may you build and submit:

```bash
bash scripts/build_submission.sh
kaggle competitions submit -c <competition-slug> -f submission.tar.gz \
  -m "SOT-1838 semantic MCTS"
kaggle competitions submissions -c <competition-slug>
```

Wait for status `complete`, then record the public score on SOT-1838. If it does not exceed 281.0,
classify the match logs and create exactly one follow-up issue for the dominant bottleneck. Generated
`cg/`, `data/`, `submission.tar.gz`, and `artifacts/` remain local or ignored.
