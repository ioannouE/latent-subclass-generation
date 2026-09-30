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
- Commands: `make test`, `make prepare` (CPU, login node).

- Everything is run on an HPC cluster. Run everything that is feasible on a login node (cpu) and if a gpu is required give me instructions on how to run it on the cluster. If required prepare a slurm script for me to run but only if it is needed. if it is for testing I can run things interactively. 
