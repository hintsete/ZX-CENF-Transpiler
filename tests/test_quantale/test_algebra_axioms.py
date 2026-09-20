"""Tests for algebra.py."""

from __future__ import annotations

import random

import pytest

from zx_cenf.quantale.algebra import QuantaleValue, big_join, top, unit


DIM = 4
N_RANDOM_CASES = 200


def _random_value(
    seed: int,
    dim: int = DIM,
    lo: float = 0.0,
    hi: float = 100.0,
) -> QuantaleValue:
   
    rng = random.Random(seed)
    return QuantaleValue(
        tuple(rng.uniform(lo, hi) for _ in range(dim))
    )




@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_tensor_is_commutative(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)

    assert a.tensor(b).coords == b.tensor(a).coords


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_tensor_is_associative(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)
    c = _random_value(seed + 200000)

    left = a.tensor(b).tensor(c)
    right = a.tensor(b.tensor(c))

    assert all(
        abs(x - y) < 1e-9
        for x, y in zip(left.coords, right.coords)
    )


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_tensor_unit_is_identity(seed):
    
    a = _random_value(seed)
    e = unit(a.dim)

    assert a.tensor(e).coords == a.coords
    assert e.tensor(a).coords == a.coords



@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_join_is_commutative(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)

    assert a.join(b).coords == b.join(a).coords


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_join_is_associative(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)
    c = _random_value(seed + 200000)

    left = a.join(b).join(c)
    right = a.join(b.join(c))

    assert left.coords == right.coords


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_join_is_idempotent(seed):
    
    a = _random_value(seed)

    assert a.join(a).coords == a.coords


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_join_is_a_valid_upper_bound(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)

    j = a.join(b)

    assert a <= j
    assert b <= j


@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_join_is_the_least_upper_bound(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)

    j = a.join(b)

    rng = random.Random(seed + 200000)
    slack = tuple(
        rng.uniform(0.0, 100.0)
        for _ in range(DIM)
    )

    upper_bound = QuantaleValue(
        tuple(
            value + extra
            for value, extra in zip(j.coords, slack)
        )
    )

    
    assert a <= upper_bound
    assert b <= upper_bound

    
    assert j <= upper_bound




@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_order_is_reflexive(seed):

    a = _random_value(seed)

    assert a <= a


def test_order_is_antisymmetric():
    
    a = QuantaleValue((1.0, 2.0, 3.0))
    b = QuantaleValue((1.0, 2.0, 3.0))

    assert a <= b
    assert b <= a
    assert a == b


def test_order_is_transitive():
    
    a = QuantaleValue((1.0, 2.0, 3.0))
    b = QuantaleValue((2.0, 3.0, 4.0))
    c = QuantaleValue((3.0, 4.0, 5.0))

    assert a <= b
    assert b <= c
    assert a <= c




@pytest.mark.parametrize("seed", range(N_RANDOM_CASES))
def test_distributivity_of_tensor_over_join(seed):
    
    a = _random_value(seed)
    b = _random_value(seed + 100000)
    c = _random_value(seed + 200000)

    left = a.tensor(b.join(c))
    right = a.tensor(b).join(a.tensor(c))

    assert all(
        abs(x - y) < 1e-9
        for x, y in zip(left.coords, right.coords)
    )




def test_dominates_requires_strict_inequality_somewhere():
    
    a = QuantaleValue((1.0, 2.0))
    b = QuantaleValue((1.0, 2.0))

    assert a <= b
    assert not a.dominates(b)


def test_dominates_basic_case():
    
    a = QuantaleValue((1.0, 2.0))
    b = QuantaleValue((1.0, 3.0))

    assert a.dominates(b)
    assert not b.dominates(a)


def test_incomparable_values_are_not_comparable():
    
    a = QuantaleValue((1.0, 5.0))
    b = QuantaleValue((5.0, 1.0))

    assert not a.comparable_to(b)
    assert not a <= b
    assert not b <= a




def test_dimension_mismatch_raises():
    
    a = QuantaleValue((1.0, 2.0))
    b = QuantaleValue((1.0, 2.0, 3.0))

    with pytest.raises(ValueError):
        a.tensor(b)


def test_join_dimension_mismatch_raises():
    a = QuantaleValue((1.0, 2.0))
    b = QuantaleValue((1.0, 2.0, 3.0))

    with pytest.raises(ValueError):
        a.join(b)


def test_negative_coordinate_rejected():
    
    with pytest.raises(ValueError):
        QuantaleValue((1.0, -0.5))


def test_nan_coordinate_rejected():
    
    with pytest.raises(ValueError):
        QuantaleValue((1.0, float("nan")))


def test_empty_coordinate_vector_rejected():
    
    with pytest.raises(ValueError):
        QuantaleValue(())




def test_top_dominates_everything_finite():
    
    t = top(3)
    v = QuantaleValue((1e9, 1e9, 1e9))

    assert v <= t
    assert not t.is_finite()
    assert v.is_finite()


def test_tensor_with_infinity_preserves_infinity():
    
    a = QuantaleValue((1.0, 2.0, 3.0))
    b = QuantaleValue((4.0, float("inf"), 6.0))

    result = a.tensor(b)

    assert result.coords == (
        5.0,
        float("inf"),
        9.0,
    )


def test_join_with_infinity_preserves_infinity():
    
    a = QuantaleValue((1.0, 2.0, 3.0))
    b = QuantaleValue((4.0, float("inf"), 2.0))

    result = a.join(b)

    assert result.coords == (
        4.0,
        float("inf"),
        3.0,
    )




def test_big_join_matches_pairwise_join():
    
    values = [
        _random_value(seed)
        for seed in range(10)
    ]

    result = big_join(values)

    manual = values[0]

    for value in values[1:]:
        manual = manual.join(value)

    assert result.coords == manual.coords


def test_big_join_single_value_returns_same_value():
    
    value = QuantaleValue((1.0, 2.0, 3.0, 4.0))

    result = big_join([value])

    assert result == value


def test_big_join_with_top_returns_top():
    
    values = [
        QuantaleValue((1.0, 2.0, 3.0, 4.0)),
        top(4),
        QuantaleValue((10.0, 20.0, 30.0, 40.0)),
    ]

    result = big_join(values)

    assert result == top(4)


def test_big_join_empty_raises():
    
    with pytest.raises(ValueError):
        big_join([])


def test_typed_empty_big_join_is_lattice_bottom():
    assert big_join([], dim=3) == unit(3)