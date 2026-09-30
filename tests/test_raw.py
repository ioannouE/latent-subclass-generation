from lsgen.data.stanford_cars import EXPECTED_COUNTS, load_annotations
from tests.conftest import requires_raw


@requires_raw
def test_annotations_counts_and_labels(raw_root):
    df = load_annotations(raw_root)  # also cross-checks labelled vs official unlabelled test annotations
    assert df.official_split.value_counts().to_dict() == EXPECTED_COUNTS
    assert df.id.is_unique and len(df) == 16185
    assert df.groupby("official_split").fine_id.nunique().to_dict() == {"train": 196, "test": 196}
    assert (df.bbox_x2 > df.bbox_x1).all() and (df.bbox_y2 > df.bbox_y1).all()
