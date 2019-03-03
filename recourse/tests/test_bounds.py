import numpy as np
from recourse.action_set import _BoundElement as BoundElement

v = np.random.rand(1000)
a = np.sort(v)
lb = np.percentile(v, 40)

# bounds

def test_absolute_bound():
    l = -1.0
    u = 10.0
    b = BoundElement(bound_type = 'absolute', lb = l, ub = u)
    assert b.lb == l
    assert b.ub == u
    assert b.bound_type == 'absolute'


def test_absolute_bound_with_values():
    l = 1.0
    u = 10.0
    values = l + np.multiply(u - l, np.random.rand(1000))
    b = BoundElement(bound_type = 'absolute', lb = l, ub = u, values = values)
    assert b.lb == l
    assert b.ub == u
    assert b.bound_type == 'absolute'


def test_absolute_bound_with_values():
    values = np.random.randn(1000)
    b = BoundElement(bound_type = 'absolute', values = values)
    assert b.lb == np.min(values)
    assert b.ub == np.max(values)
    assert b.bound_type == 'absolute'


def test_percentile_bound():
    l = 5.0
    u = 95.0
    values = np.random.rand(1000)
    b = BoundElement(bound_type = 'percentile', lb = l, ub = u, values = values)
    assert np.isclose(b.lb, np.percentile(values, l))
    assert np.isclose(b.ub, np.percentile(values, u))
    assert b.bound_type == 'percentile'



# bounds