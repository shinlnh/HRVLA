# Local storage cleanup ledger

This ledger records destructive local cleanup so published evidence remains
auditable and recoverable. Source code, runtime dependencies, selected final
checkpoints, prepared datasets, and all tracked reports/plots were preserved.

## 2026-09-15 cleanup

Before deletion, the Hugging Face API and local LFS content were checked against
immutable revisions:

- model repository `shin0412/HRVLA` at
  `50ea75ab7972509c2f6827190a965484a2057bcc`;
- dataset repository `shin0412/HRVLA` at
  `a745b9eba09f2beac02f6f018f3c9c369b0dc5a5`;
- all 18 large shards from the six multi-seed step-300 checkpoints matched remote
  LFS SHA-256 and byte size;
- all six shards from the two original selected step-300 checkpoints matched;
- the remote model revision contained the six checkpoint trees and nine compact
  paper-evidence files; the dataset revision contained all 858 prepared split
  files.

Deleted from `_artifacts/retraining`:

- `paper-seeds/` — approximately 143 GiB. The six selected final checkpoints and
  compact metrics are recoverable from the model revision above. Intermediate
  step-100/200 and duplicate root weights were not selected publication
  artifacts; they are recoverable only by rerunning the locked training recipe.
- `st-smoke/` and `st-batch4-probe/` — approximately 24 GiB. These were
  non-selected development probes, not claim-bearing evidence, and are
  reproducible from the committed configuration/history.
- ST production step-100, step-200, and duplicate root model shards —
  approximately 18 GiB. Selected step-300 remains local and on Hugging Face.
- STR production step-100, step-200, and duplicate root model shards —
  approximately 18 GiB. Selected step-300 remains local and on Hugging Face.

Approximately **202 GiB** was reclaimed. Free filesystem space increased from
about 189 GiB to about 391 GiB.

After the three-way HumanoidArena split passed membership, pixel, and
state/action round-trip validation, the superseded 335 MiB two-way derived view
was removed and the validated 70/10/20 view took its canonical path. The
immutable source dataset remains local and at the locked Hugging Face revision;
the builder, final manifest hash, validation JSON, and plot are tracked, so the
deleted derived view is reproducible and is not claim evidence.

## Explicitly retained

- `_artifacts/retraining/.../checkpoint-300` for the selected ST-RT and STR-RT
  production models;
- `_artifacts/GR00T-N1.7-3B`, HumanoidArena's released task checkpoints,
  released dataset, tokenizer, SONIC policy, Isaac Lab, and vendor runtimes;
- prepared local training splits required for reproducibility;
- every tracked JSON/JSONL/CSV/log checksum, chart, screenshot, and video;
- all 51 completed episode JSONs and videos from the interrupted HumanoidArena
  run;
- `.vscode/` and every user-owned worktree.

The deleted files were removed from local storage rather than moved to trash.
Published final artifacts can be downloaded again by immutable revision. The
development-only and intermediate checkpoints require deterministic reruns.
