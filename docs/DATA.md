# Data protocol (Step 1, Milestone 1)

Built by `make prepare` (`scripts/prepare_data.py --config configs/data.yaml`). CPU only; runs on a login node.

## Raw data (read-only)
`raw_root = /nvme/h/eioannou/data_p315/Stanford_Cars/kaggle/` (Kaggle mirror `eduardo4jesus/stanford-cars-dataset`).

| Path | Content |
|---|---|
| `car_devkit/` | empty (artefact of the Kaggle mirror) |
| `devkit/cars_meta.mat` | 196 class names |
| `devkit/cars_train_annos.mat` | 8,144 records: bbox, class, fname |
| `devkit/cars_test_annos.mat` | 8,041 records: bbox, fname (no class) |
| `devkit/cars_test_annos_withlabels.mat` | 8,041 records incl. class. Third-party copy from github.com/jhpohovey/StanfordCars-Dataset; sha256 `790f75be…` equals the copy in `Stanford_Cars/torchvision_layout/`. Verified record-by-record (fname + bbox, same order) against the official unlabelled file on every run |
| `cars_train/cars_train/*.jpg`, `cars_test/cars_test/*.jpg` | images, one folder level deeper than the folder name suggests |

Verified on every run: 16,185 images (8,144 / 8,041), 196 classes in each split, one bbox per image, every
annotation points to a readable image, all bboxes within `[1, W] x [1, H]`; 34 greyscale images. SHA256 of every raw
file: `data/manifests/raw_manifest.json` (its `manifest_hash` is the "dataset manifest hash" recorded in outputs).

## Labels
- Coarse = make (49), hidden subclass = official class (make-model-year, 196), secondary = make-model.
- Parsing: `lsgen/data/hierarchy.py`. Makes are the first token except the reviewed multi-word makes
  AM General, Aston Martin, Land Rover. "AM General" and "HUMMER" are distinct official makes; "Ram" is distinct from "Dodge".
- Make-model = exact model string with the year removed (body type kept), e.g. "Audi S4 Sedan" merges 2007 + 2012.
  This merges only 7 pairs, giving 189 make-model classes.
- `data/hierarchy.csv` (fine_id = devkit class - 1, make_id, model_id, year, K_c), `data/make_model.csv`.
- Ids (`make_id`, `model_id`) are assigned in order of first appearance in `cars_meta.mat`.

## Splits (`data/splits/*.txt`, one image id per line; ids are `{official_split}_{fname stem}`)
| Split | Definition | n |
|---|---|---|
| train | official train minus val | 7,329 |
| val | 10% of official train, `StratifiedShuffleSplit` by fine_id, seed 0 | 815 |
| test | official test; never used for tuning or model selection | 8,041 |
| test_A / test_B | stratified (fine_id) halves of test, seed 0 | 4,020 / 4,021 |

Fine labels are used here **only** to stratify. Training datasets (`lsgen/data/datasets.py`) return `(image, make_id)` only.

## Preprocessing (docs/PLAN.md A2b)
Derived files live outside git in `derived_root = /nvme/h/eioannou/data_p315/Stanford_Cars/lsgen_derived/`:

| Path | Content |
|---|---|
| `bbox15_{128,256}/{id}.png`, `full_cc_{128,256}/{id}.png` | crops (lossless PNG, compress_level 6, no metadata) |
| `metadata.parquet` | one row per image (see below) |
| `manifest_derived.json` | SHA256 of every derived file + provenance (summary committed in `data/manifests/derived_summary.json`) |
| `raw_index.parquet`, `crops_meta.parquet` | intermediate stage outputs |
| `duplicates/` | pHash codes, SSCD embeddings of raw images, test->train NN table |
| `weights/` | SSCD `disc_mixup` TorchScript |

`metadata.parquet` columns: id, path (bbox15_128 file), rel_path (raw), official_split, split, test_half, make_id, fine_id,
model_id, bbox_x1..y2 (devkit, 1-indexed inclusive), bbox_clipped, crop_x0..y1 (square window in 0-indexed original pixel
coords; extends outside the image where padded), side (window side in original px), scale_128/scale_256, pad_l/t/r/b
(original px), pad_fraction (padded area / window area), margin_used (= side / max(bbox w, h)), was_grayscale, raw_mode,
orig_W, orig_H, raw_sha256, near_duplicate, n_near_duplicates.

Determinism: a random subset of 300 images is regenerated and must match the written files byte for byte.

## Near-duplicate audit
Raw official-train vs official-test images: 64-bit pHash (Hamming <= 8) OR SSCD disc_mixup cosine > 0.5 (small_288
transform). Candidate pairs: `data/duplicates.csv`; flagged images are marked in metadata (`near_duplicate`), never removed.
The per-test-image nearest-train SSCD similarity (`duplicates/sscd_nn_test_to_train.parquet`) is the null distribution for G7.

## Native pipelines
`lsgen/data/native.py` (`native_finegan`, `native_c3gan`) only for reproducing published numbers; never used in comparison tables.
