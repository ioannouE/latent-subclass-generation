"""Error analysis of frozen DINOv3 subclass neighbours within each coarse class (diagnostic only, no training). CPU only, reads cached
embeddings:
    sbatch slurm/error_analysis.sh                                                      # make / make-model
    CONFIG=configs/error_analysis_maskcon.yaml sbatch slurm/error_analysis.sh           # MaskCon: body type / 196 classes
Cases (fixed rule, lsgen/eval/neighbours.py): the n different-subclass pairs with the highest DINO cosine (false), the n same-subclass
pairs with the lowest (missed), and for each failure query its highest-ranked same-subclass neighbour (control).
Output: <reports_root>/<report_dir>/{pairs.csv, lowest_auroc_<coarse>s.csv, summary.json} (default error_analysis/; pairs.csv has empty
columns for manual tags); <reports_root>/<report_dir>/neighbours.html (the pairs side by side; images linked by relative path to the derived crops and the
raw photos, nothing copied; serve with `python -m http.server --directory /nvme/h/eioannou` to view).
"""
import argparse
import html
import json
import logging
import os
from pathlib import Path

import pandas as pd

from lsgen.data.labels import scheme as label_scheme
from lsgen.data.splits import load_split
from lsgen.eval.neighbours import controls, failure_cases, make_aurocs, pair_table, subclass_labels
from lsgen.eval.report import exclusion_note, write_results
from lsgen.eval.representation import l2norm
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("error_analysis")
TAGS = ["same_view", "same_colour", "similar_background", "genuinely_similar_models", "label_error_or_duplicate"]
GROUPS = {"false": "False neighbours: highest DINO cosine, different subclass",
          "missed": "Missed neighbours: lowest DINO cosine, same subclass",
          "control": "Controls: each failure query's highest-ranked same-subclass neighbour"}


def load_embeddings(derived, encoder, cfg, ids):
    files, x = read_csv(derived / "features" / f"{encoder}_{cfg['crop_size']}" / f"{cfg['split']}.csv")
    return l2norm(pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy())


def describe(cases, group, ids, make_name, sub_name, sscd_threshold, coarse="make"):
    """Index-based case rows -> the CSV rows (ids, names, scores, near-duplicate flag, empty tag columns)."""
    a, b = cases.a.to_numpy(int), cases.b.to_numpy(int)
    out = pd.DataFrame({"group": group, coarse: make_name[a], "query": [ids[i] for i in a], "partner": [ids[i] for i in b],
                        "query_subclass": sub_name[a], "partner_subclass": sub_name[b], "same_subclass": cases.same.to_numpy(),
                        "dino": cases.dino.round(4).to_numpy(), "dreamsim": cases.dreamsim.round(4).to_numpy(),
                        "rank_query_to_partner": cases.rank_ab.to_numpy(int), "rank_partner_to_query": cases.rank_ba.to_numpy(int),
                        "sscd": cases.sscd.round(4).to_numpy()})
    out["possible_near_duplicate"] = out.sscd >= sscd_threshold
    return out.assign(**dict.fromkeys(TAGS, ""))


def html_page(pairs, low, crop_dir, raw_paths, page_dir, note, sscd_threshold, coarse="make"):
    """Lowest-AUROC coarse classes, then each group's pairs side by side with their scores; each crop links to its raw photo (relative paths)."""
    e = html.escape
    c = coarse.replace("_", " ")

    def img(i):
        return (f"<a href='{e(os.path.relpath(raw_paths[i], page_dir))}'>"
                f"<img src='{e(os.path.relpath(crop_dir / f'{i}.png', page_dir))}' loading='lazy'></a>")
    parts = [f"<h1>DINOv3 subclass neighbours: error analysis</h1><p>{e(note)}. Cosines within the {c}; rank = position in the "
             f"other image's DINO neighbour list within the {c} (1 = nearest). Near-duplicate flag: SSCD cosine &ge; {sscd_threshold}.</p>",
             f"<h2>{e(c.capitalize())}s with the lowest DINO AUROC</h2>", low.to_html(index=False, float_format="%.3f", classes="t")]
    for group, title in GROUPS.items():
        g = pairs[pairs.group == group]
        parts.append(f"<h2>{e(title)} ({len(g)})</h2><div class='grid'>")
        for r in g.itertuples():
            dup = " <b class='dup'>possible near-duplicate</b>" if r.possible_near_duplicate else ""
            link = f" &middot; for {e(r.control_for)}" if r.control_for else ""
            parts.append(
                f"<div class='pair'><div class='imgs'>{img(r.query)}{img(r.partner)}</div><div class='s'><b>{e(r.pair_id)}</b>{link} &middot; "
                f"{e(getattr(r, coarse))}<br>query {e(r.query)}: {e(r.query_subclass)}<br>partner {e(r.partner)}: {e(r.partner_subclass)}<br>"
                f"DINO {r.dino:.3f} &middot; DreamSim {r.dreamsim:.3f} &middot; SSCD {r.sscd:.3f}{dup}<br>"
                f"rank q&rarr;p {r.rank_query_to_partner} &middot; p&rarr;q {r.rank_partner_to_query}</div></div>")
        parts.append("</div>")
    style = ("<style>:root{--bg:#fff;--fg:#1a1a1a;--line:#ddd;--dup:#b00020}@media (prefers-color-scheme:dark){:root{--bg:#161616;"
             "--fg:#e6e6e6;--line:#333;--dup:#ff6b81}}body{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;"
             "margin:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}.pair{border:1px solid "
             "var(--line);padding:8px}.imgs{display:flex;gap:4px}.imgs a{width:50%}.imgs img{width:100%;height:auto}.s{margin-top:6px;line-height:1.4}"
             ".dup{color:var(--dup)}.t{border-collapse:collapse}.t td,.t th{border:1px solid var(--line);padding:2px 8px}</style>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Neighbour error analysis ({e(c)})</title>{style}</head><body>{''.join(parts)}</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/error_analysis.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    sscd_threshold = dcfg["duplicates"]["sscd_threshold"]
    scheme = cfg.get("labels", "make")
    coarse = "body_type" if "coarse" in label_scheme(scheme) else "make"   # name of the coarse class (configs/labels.yaml)
    ids = load_split(splits_dir, cfg["split"], labels=scheme)
    make, make_name, sub, sub_name = subclass_labels(derived, dcfg["data_root"], ids, scheme)
    emb = {k: load_embeddings(derived, cfg[k], cfg, ids) for k in ("dino", "dreamsim", "sscd")}

    pairs = pair_table(make, sub, emb)
    false, missed = failure_cases(pairs, cfg["n_cases"])
    rows = []
    for group, cases in (("false", false), ("missed", missed)):
        d = describe(cases, group, ids, make_name, sub_name, sscd_threshold, coarse)
        rows.append(d.assign(pair_id=[f"{group}_{k:02d}" for k in range(len(d))], control_for=""))
        ctrl, owner = controls(pairs, cases)
        c = describe(ctrl, "control", ids, make_name, sub_name, sscd_threshold, coarse)
        rows.append(c.assign(pair_id=[f"control_{group}_{k:02d}" for k in owner], control_for=[f"{group}_{k:02d}" for k in owner]))
    table = pd.concat(rows, ignore_index=True)
    table = table[["pair_id", "control_for", *table.columns.drop(["pair_id", "control_for"])]]
    low = make_aurocs(make, sub, emb["dino"]).head(cfg["n_lowest_auroc"])
    low.insert(1, f"{coarse}_name", make_name[[list(make).index(c) for c in low.make]])
    low = low.rename(columns={"make": f"{coarse}_id", "auroc": "auroc_dino"})

    coverage = {g: {"pairs": int((table.group == g).sum()), f"{coarse}s": int(table[table.group == g][coarse].nunique()),
                    "distinct_images": int(pd.unique(table[table.group == g][["query", "partner"]].values.ravel()).size),
                    "possible_near_duplicates": int(table[table.group == g].possible_near_duplicate.sum())} for g in GROUPS}
    out_dir = Path(dcfg["reports_root"]) / cfg.get("report_dir", "error_analysis")
    write_results(out_dir, {"pairs": table, f"lowest_auroc_{coarse}s": low}, cfg, cfg["seed"], dataset_hash, splits_dir,
                  n_images=len(ids), **{f"n_{coarse}s": int(pairs.make.nunique())}, n_candidate_pairs=len(pairs), sscd_threshold=sscd_threshold,
                  coverage=coverage)
    excluded = exclusion_note(splits_dir, labels=scheme)["excluded_conflicting_images"][cfg["split"]]
    note = (f"{cfg['split']} split, {len(ids)} images after excluding {excluded} conflicting images, {pairs.make.nunique()} {coarse.replace('_', ' ')}s with "
            f">= 2 subclasses; {cfg['dino']}, {cfg['dreamsim']}, {cfg['sscd']} on {cfg['crop']}_{cfg['crop_size']} crops; git {git_hash()}")
    page = out_dir / "neighbours.html"
    raw = pd.read_parquet(derived / "raw_index.parquet", columns=["id", "rel_path"]).set_index("id").rel_path
    raw_paths = {i: Path(dcfg["raw_root"]) / raw[i] for i in pd.unique(table[["query", "partner"]].values.ravel())}
    page.write_text(html_page(table, low, derived / f"{cfg['crop']}_{cfg['crop_size']}", raw_paths, out_dir, note, sscd_threshold,
                                coarse))
    log.info("%s\ncoverage: %s\nlowest DINO AUROC:\n%s\nwrote %s and %s", note, coverage, low.round(3).to_string(index=False), out_dir, page)


if __name__ == "__main__":
    main()
