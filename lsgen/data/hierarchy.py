"""Class-name hierarchy: make (coarse, 49) > make-model (years merged) > make-model-year (fine, 196).

Class names have the form "<Make> <Model ...> <Year>". Makes are the first token, except for the
multi-word makes below. The full list of 49 makes was reviewed by hand against cars_meta.mat:
  AM General, Acura, Aston Martin, Audi, BMW, Bentley, Bugatti, Buick, Cadillac, Chevrolet,
  Chrysler, Daewoo, Dodge, Eagle, FIAT, Ferrari, Fisker, Ford, GMC, Geo, HUMMER, Honda, Hyundai,
  Infiniti, Isuzu, Jaguar, Jeep, Lamborghini, Land Rover, Lincoln, MINI, Maybach, Mazda, McLaren,
  Mercedes-Benz, Mitsubishi, Nissan, Plymouth, Porsche, Ram, Rolls-Royce, Scion, Spyker, Suzuki,
  Tesla, Toyota, Volkswagen, Volvo, smart
Notes: "AM General" (Hummer SUV 2000) and "HUMMER" (H2/H3T) are distinct official makes; "Ram"
(C/V Cargo Van) is distinct from "Dodge"; hyphenated makes are single tokens.
Make-model = exact match of the model string (body type included), years merged.
"""
import re

import pandas as pd

MULTIWORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
N_MAKES = 49
_YEAR = re.compile(r"^(19|20)\d\d$")


def parse_class_name(name):
    """'Aston Martin V8 Vantage Convertible 2012' -> ('Aston Martin', 'V8 Vantage Convertible', 2012)."""
    tokens = name.split()
    if len(tokens) < 3 or not _YEAR.match(tokens[-1]):
        raise ValueError(f"cannot parse class name {name!r}: expected '<make> <model> <year>'")
    make = next((m for m in MULTIWORD_MAKES if name.startswith(m + " ")), tokens[0])
    model = name[len(make):].rsplit(" ", 1)[0].strip()
    if not model:
        raise ValueError(f"empty model in {name!r}")
    return make, model, int(tokens[-1])


def build_hierarchy(class_names):
    """DataFrame indexed by fine_id with make/model/year and integer ids (in order of first appearance)."""
    rows = [dict(fine_id=i, class_name=n, **dict(zip(("make", "model", "year"), parse_class_name(n))))
            for i, n in enumerate(class_names)]
    h = pd.DataFrame(rows)
    h["make_model"] = h.make + " " + h.model
    h["make_id"] = pd.factorize(h.make)[0]
    h["model_id"] = pd.factorize(h.make_model)[0]
    h["K_c"] = h.groupby("make").fine_id.transform("count")
    if h.make.nunique() != N_MAKES:
        raise ValueError(f"expected {N_MAKES} makes, got {h.make.nunique()}: {sorted(h.make.unique())}")
    return h


def make_model_table(h):
    return (h.groupby(["model_id", "make_model", "make", "make_id", "model"], as_index=False)
             .agg(n_years=("year", "count"), years=("year", lambda y: " ".join(map(str, sorted(y)))),
                  fine_ids=("fine_id", lambda f: " ".join(map(str, f)))))


def kc_per_make(h):
    return h.groupby(["make_id", "make"]).fine_id.count().rename("K_c").reset_index()
