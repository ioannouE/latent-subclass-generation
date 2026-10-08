"""Label schemes. "make" (default, everything done so far): coarse = make (49), hidden subclass = make-model (189).
"maskcon": the MaskCon protocol (Feng & Patras, CVPR 2023): coarse = 8 body types, hidden subclass = the 196 official classes.
The mapping is data/maskcon_coarse.csv (copied from the official `_cars_mapping`). Either way the returned table has the columns
`make_id` (= the coarse label) and `fine_id` (= the hidden subclass), so metrics, training and plots need no other change."""
from pathlib import Path

import pandas as pd

SCHEMES = ("make", "maskcon")


def load_labels(derived_root, data_root, scheme="make"):
    """id-indexed DataFrame: make_id (coarse), fine_id (subclass), coarse_name, fine_name. Conflicting images are not dropped here."""
    if scheme not in SCHEMES:
        raise ValueError(f"label scheme must be one of {SCHEMES}")
    meta = pd.read_parquet(Path(derived_root) / "metadata.parquet", columns=["id", "make_id", "fine_id", "class_id"]).set_index("id")
    hierarchy = pd.read_csv(Path(data_root) / "hierarchy.csv").set_index("class_id")
    if scheme == "make":
        meta["coarse_name"] = hierarchy.make.loc[meta.class_id].to_numpy()
        meta["fine_name"] = hierarchy.make_model.loc[meta.class_id].to_numpy()
    else:
        body = pd.read_csv(Path(data_root) / "maskcon_coarse.csv").set_index("class_id")
        meta["make_id"], meta["coarse_name"] = body.body_type_id.loc[meta.class_id].to_numpy(), body.body_type.loc[meta.class_id].to_numpy()
        meta["fine_id"], meta["fine_name"] = meta.class_id, hierarchy.class_name.loc[meta.class_id].to_numpy()
    return meta.drop(columns="class_id")
