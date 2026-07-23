# Zero/Sol GPU experiment handoff

`configs/experiments/zero-sol-v1.json` is the versioned handoff from Sol deck preselection to the
joint Zero/Sol experiment. Before a GPU run:

1. Generate or resume the deterministic shortlist with `ptcg-agent preselect-decks`.
2. Review the baseline comparison, hard-opponent metrics, legality, diversity, seed, budget, and
   candidate hash in `artifacts/sot-1890-deck-shortlist.json`.
3. Copy the promoted shortlist and runtime dataset to the Zero experiment input location.
4. Update and commit their SHA-256 values in Zero's `configs/experiments/zero-sol-v1.json`.
5. Run Zero's documented `ptcg-agent-zero experiment` command without `--dry-run`.

Do not promote a result when the shortlist hash differs from the committed config or when the
experiment artifact manifest does not contain all four stages (`train`, `search`, `eval`, `package`).
