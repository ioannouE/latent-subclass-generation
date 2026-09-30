# latent-subclass-generation
Spec: docs/PLAN.md (Part A = design, Part B = Step 1 prompt). Read it before any task.

## Hard rules
- Fine labels only inside lsgen/eval/ and split/episode construction; training datasets never return them.
- `test` split is never used for tuning or model selection.
- Raw data at /nvme/h/eioannou/data_p315/Stanford_Cars/kaggle/ is read-only; never commit data, features, checkpoints or images.
- Evaluators must be disjoint from any encoder a method uses.
- Work one milestone at a time; stop and report after each. Don't tune a metric to make a control pass.
- Branch per milestone (step1/m1-data, ...), commit and push at the end of each.