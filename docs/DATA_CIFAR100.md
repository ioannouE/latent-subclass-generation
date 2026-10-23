# CIFAR-100 benchmark: data

Second benchmark next to Stanford Cars (docs/DATA.md), with the same pipeline: `configs/data_cifar100.yaml` plays the role of
`configs/data.yaml`, and any config that sets `data_config: configs/data_cifar100.yaml` and `labels: cifar100` runs on CIFAR-100.

## Labels
- Coarse = the 20 official superclasses (`make_id`), hidden subclass = the 100 classes (`fine_id` = `class_id` = official fine label).
  Both come with the official files; no mapping is ours.
- Every superclass has exactly 5 classes (K_c = 5) and every class 500 train / 100 test images: no K_c = 1 superclasses (R5's K-hat error has
  nothing to score) and no rare subclasses.
- Superclasses are semantic, not visual (e.g. "vehicles 1" vs "vehicles 2", "medium-sized" vs "small mammals").
- `data/cifar100/hierarchy.csv` has the columns of the Stanford Cars hierarchy (`make` = superclass name, `make_model` = class name), so
  `lsgen.data.labels.load_labels` and everything downstream read it unchanged.

## Raw data (read-only)
`raw_root = /nvme/h/eioannou/data_p315/CIFAR100/raw/`: the official python version (Krizhevsky 2009). Download once, from the login node:

    mkdir -p /nvme/h/eioannou/data_p315/CIFAR100/raw && cd /nvme/h/eioannou/data_p315/CIFAR100/raw
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    echo "eb9058c3a382ffc7106e4002c42a8d85  cifar-100-python.tar.gz" | md5sum -c && tar xzf cifar-100-python.tar.gz
    chmod -R a-w /nvme/h/eioannou/data_p315/CIFAR100/raw

Verified on every run: md5 of `cifar-100-python/{train,test,meta}` (torchvision's values), 50,000 / 10,000 images, 500 / 100 per class, every class
in exactly one superclass, 20 x 5. SHA256 of the three files: `data/cifar100/manifests/raw_manifest.json` (its `manifest_hash` is the dataset
manifest hash recorded in outputs).

## Images
Native 32 px, no crop or resize: `derived_root/native_32/{id}.png` (lossless, compress_level 6), ids `{official_split}_{row:05d}` in the row order
of the official file. Configs refer to them as crop `native_32`; each encoder resizes them to its own input size (224 for DINOv3), the same
path for real and generated images. 300 random images are re-encoded on every run and must match byte for byte and pixel for pixel.

`derived_root = /nvme/h/eioannou/data_p315/CIFAR100/lsgen_derived/`: `native_32/`, `raw_index.parquet`, `metadata.parquet`, `png_shas.json`,
`duplicates/sscd_embeddings.npy`. `metadata.parquet` columns: id, path, official_split, row, split, test_half, make_id, fine_id, class_id,
raw_sha256 (of the pixels), near_duplicate, n_near_duplicates, exclude.

## Splits (`data/cifar100/splits/*.txt`)
| Split | Definition | n |
|---|---|---|
| train | official train minus val | 45,000 |
| val | 10% of official train, `StratifiedShuffleSplit` by class, seed 0 | 5,000 |
| test | official test; never used for tuning or model selection | 10,000 |
| test_A / test_B | stratified halves of test, seed 0 | 5,000 / 5,000 |

Counts are before excluding conflicting images (below); `load_split(..., labels="cifar100")` drops them.

## Conflicting images
CIFAR-100 has train-test near-duplicates (Barz & Denzler 2020, "ciFAIR"). As for the `maskcon_new` scheme on Stanford Cars, every pair of the
60,000 images (any splits) is compared: pixel-identical, or SSCD disc_mixup cosine >= `sscd_threshold` (32 px upsampled to 288). Images in such
a pair with a **different class** are excluded from every experiment (`data/cifar100/splits/exclude.txt`, `metadata.exclude`); same-class
pairs are marked (`near_duplicate`), not removed. pHash is not used. All pairs with cosine >= `audit_min_sim` (0.65) are kept in
`data/cifar100/duplicates_all.csv`.

**Threshold 0.75 (decision 23 Oct 2026), the Stanford Cars value**, after reading the per-band counts and pair grids in
`reports/cifar100/data_summary.md` (the grids show dataset images and are not committed). Different-class pairs:

| SSCD band | pairs | of which fruit-vs-fruit | what they are |
|---|---|---|---|
| [0.70, 0.75) | 98 | 94 | different apples / pears / oranges on white backgrounds: not duplicates |
| [0.75, 0.85) | 24 | 12 | the 12 others are true duplicates (shrew/mouse, porcupine/shrew, leopard/tiger, snake/worm, shark/whale, streetcar/train); the fruit ones mostly not |
| >= 0.85, or pixel-identical | 33 | 1 | all true duplicates (17 pixel-identical) |

0.85 would keep the 12 true cross-class duplicates in [0.75, 0.85); 0.75 catches them at the cost of about a dozen fruit pairs (~20 images
excluded unnecessarily). Result: 57 pairs, **90 images excluded** (train 68, val 5, test 17); the 1,378 same-class train-test near-duplicates
at >= 0.75 are marked, not removed.

## Run
Data: `sbatch slurm/prepare_cifar100.sh` (stages raw, splits, duplicates, finalize, report; SSCD needs the GPU).

Representation table (same scripts as Stanford Cars, CIFAR-100 configs; results in `reports/cifar100/repr/<encoder>_32/`):

| Step | Command | Needs |
|---|---|---|
| frozen features | `CONFIG=configs/features_cifar100.yaml sbatch slurm/extract_features.sh` | data |
| frozen R1-R5 | `CONFIG=configs/eval_repr_cifar100.yaml sbatch slurm/eval_representation.sh` | frozen features |
| SupCon, SimCLR, SupCon+SimCLR | `CONFIG=configs/repr_baselines_cifar100.yaml METHODS="supcon simclr supcon_simclr" sbatch slurm/train_repr.sh` | data |
| LeJEPA, MaskCon, FALCON | `CONFIG=configs/repr_baselines_cifar100.yaml METHODS="lejepa maskcon falcon" sbatch slurm/train_repr.sh` | frozen features (FALCON's neighbours) |
| GEORGE | `CONFIG=configs/repr_george_cifar100.yaml sbatch slurm/train_george.sh` | data |
| baselines R1-R5 | `CONFIG=configs/eval_repr_cifar100_baselines.yaml sbatch slurm/eval_representation.sh` | all baselines trained |

NHP is not run: its nuisance views need object bounding boxes.
