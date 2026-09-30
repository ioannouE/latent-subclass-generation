from pathlib import Path

import pytest

from lsgen.utils import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "data.yaml")
requires_raw = pytest.mark.skipif(not Path(CFG["raw_root"]).exists(), reason=f"raw data not found at {CFG['raw_root']}")


@pytest.fixture
def raw_root():
    return CFG["raw_root"]
