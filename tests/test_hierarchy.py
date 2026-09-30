import pytest

from lsgen.data.hierarchy import build_hierarchy, parse_class_name
from tests.conftest import requires_raw


@pytest.mark.parametrize("name,expected", [
    ("AM General Hummer SUV 2000", ("AM General", "Hummer SUV", 2000)),
    ("Aston Martin V8 Vantage Convertible 2012", ("Aston Martin", "V8 Vantage Convertible", 2012)),
    ("Land Rover Range Rover SUV 2012", ("Land Rover", "Range Rover SUV", 2012)),
    ("Mercedes-Benz S-Class Sedan 2012", ("Mercedes-Benz", "S-Class Sedan", 2012)),
    ("Ram C/V Cargo Van Minivan 2012", ("Ram", "C/V Cargo Van Minivan", 2012)),
    ("smart fortwo Convertible 2012", ("smart", "fortwo Convertible", 2012)),
    ("HUMMER H2 SUT Crew Cab 2009", ("HUMMER", "H2 SUT Crew Cab", 2009)),
])
def test_parse_class_name(name, expected):
    assert parse_class_name(name) == expected


@pytest.mark.parametrize("bad", ["Audi 2012", "Audi TT Coupe", "BMW M3 Coupe 12"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_class_name(bad)


@requires_raw
def test_real_hierarchy(raw_root):
    from lsgen.data.stanford_cars import load_class_names
    h = build_hierarchy(load_class_names(raw_root))
    assert len(h) == 196 and h.make.nunique() == 49
    assert set(h.loc[h.make.str.contains(" "), "make"]) == {"AM General", "Aston Martin", "Land Rover"}
    assert h.groupby("make").K_c.first().sum() == 196
    assert h.model_id.nunique() <= 196 and (h.groupby("make_model").make.nunique() == 1).all()
