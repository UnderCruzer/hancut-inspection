import pytest

from hancut.eval import dense


def _rows(*triples):
    return [{"image_id": i, "item": it, "y_true": y, "score": s} for i, it, y, s in triples]


def test_class_maxima_per_class():
    assert dense.class_maxima([[0.1, 0.4, 0.2], [0.0, 0.0, 0.3]]) == [0.4, 0.3]


def test_dominance_catches_dense_below_nms():
    d = _rows(("a", "Gun", 1, 0.30), ("a", "Knife", 0, 0.10))
    n = _rows(("a", "Gun", 1, 0.0), ("a", "Knife", 0, 0.20))
    assert dense.dominance_violations(d, n) == [("a", "Knife", 0.10, 0.20)]
    with pytest.raises(ValueError):
        dense.dominance_violations(_rows(("z", "Gun", 1, 0.1)), n)


def test_zero_rate_counts_only_positives():
    rows = _rows(("a", "Gun", 1, 0.0), ("b", "Gun", 1, 0.2), ("c", "Gun", 0, 0.0), ("a", "Knife", 1, 0.5))
    assert dense.zero_rate_by_item(rows) == {"Gun": (1, 2), "Knife": (0, 1)}


def test_agreement_within_tolerance():
    a = _rows(("a", "Gun", 1, 0.5), ("b", "Gun", 0, 0.1))
    b = _rows(("a", "Gun", 1, 0.5004), ("b", "Gun", 0, 0.3))
    assert dense.agreement(a, b) == (1, 2)
