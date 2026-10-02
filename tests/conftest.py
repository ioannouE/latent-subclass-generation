import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # tiny test matrices are much slower with dozens of BLAS threads

import pytest  # noqa: E402

from lsgen.utils import REPO_ROOT, load_config  # noqa: E402

CFG = load_config(REPO_ROOT / "configs" / "data.yaml")
requires_raw = pytest.mark.skipif(not Path(CFG["raw_root"]).exists(), reason=f"raw data not found at {CFG['raw_root']}")


@pytest.fixture
def raw_root():
    return CFG["raw_root"]
