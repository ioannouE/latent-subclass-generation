# latent-subclass-generation
Spec: docs/PLAN.md (Part A = design, Part B = Step 1 prompt). Read it before any task.

## Hard rules
- Fine labels only inside lsgen/eval/ and split/episode construction; training datasets never return them.
- `test` split is never used for tuning or model selection.
- Raw data at /nvme/h/eioannou/data_p315/Stanford_Cars/kaggle/ is read-only; never commit data, features, checkpoints or images.
- Evaluators must be disjoint from any encoder a method uses.
- Work one milestone at a time; stop and report after each. Don't tune a metric to make a control pass.
- Work directly on `main` (no feature branches); commit and push at the end of each milestone. Keep `main` passing `make test`.

## Environment
- Conda env `diffusion` (`conda run -n diffusion ...`); package installed with `pip install -e ".[dev]"`.
- Config: configs/data.yaml. Derived data (crops, parquet, manifests, weights) lives in
  /nvme/h/eioannou/data_p315/Stanford_Cars/lsgen_derived/ (never in git). Small frozen artefacts
  (hierarchy, splits, duplicates, manifest summaries) live in repo `data/` and are committed.
- Everything runs on an HPC cluster. Don't run pipeline jobs (data prep, feature extraction, training, evaluation)
  yourself: write GPU-ready code plus a Slurm script in `slurm/`, and tell the user the exact command to run from a
  GPU node (e.g. `sbatch slurm/prepare_data.sh`). Quick checks like `make test` are fine on the login node; for
  interactive testing the user can run things themselves.
- Slurm convention (see slurm/prepare_data.sh): `slurm/<name>.sh`, `#SBATCH` header (partition rtx, `-A p315`,
  `--output=%x.%j.out`), then a few plain shell variables for what to run. Keep it short; settings live in configs/*.yaml.

## Code style (applies to every script and module)
- Code like a researcher in a well-maintained CVPR-paper repository: minimal, necessary, readable.
- Only make the changes and additions the task needs. No speculative features, extra flags, override systems,
  wrappers or abstractions "for later". Prefer editing existing code over adding new files.
- Small pure functions, clear names, short docstrings where the intent isn't obvious; configs in YAML, no
  hard-coded paths elsewhere; seeded and deterministic; outputs record git hash, config and seed.
- Tests cover correctness-critical logic (metrics, crops, splits, leakage rules) with small synthetic cases, not
  plumbing.
- Don't overcomplicate: if a simpler solution does the job, use it.
