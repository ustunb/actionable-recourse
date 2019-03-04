from recourse.builder import RecourseBuilder
from recourse.action_set import ActionSet
import numpy as np
from recourse.paths import *

class MyActionSet(ActionSet):
    def __init__(self):
        ## default values for "X" (dataset) and "names" (column names).
        X = np.array([[1, 2, 3], [2, 3, 4]])
        names = ['a', 'b', 'c']

        ## initialize.
        super().__init__(X, names=names)


def test_actionset_fake_data():
    a = MyActionSet()

    ## check that the column-names are being stored correctly
    assert a._names == ['a', 'b', 'c']

    ## check that the action set is not aligned
    assert a.aligned == False

    ## test alignment
    coefficients = [1, 2, 3]
    a.align(coefficients)
    assert a.aligned == True

    ## test action grid
    x = [1, 1, 1]
    actions, percentiles = a.feasible_grid(x=x)
    assert actions['a'].tolist() == [0, 1]
    assert actions['b'].tolist() == [0, 1, 2]
    assert actions['c'].tolist() == [0, 2, 3]

    assert np.isclose(percentiles['a'], [.5, 1], atol=1e-6).all()
    assert np.isclose(percentiles['b'], [0, .5, 1], atol=1e-6).all()
    assert np.isclose(percentiles['c'], [0, .5, 1], atol=1e-6).all()


def test_recourse_builder_cplex_fake_data():
    """Base test. Tests that the factory object instantiates a
    _RecourseCPLEX module and that it has correct initialization attributes."""

    a = MyActionSet()
    x = [1, 1, 1]
    coefficients = [1, 2, 3]
    t_cplex = RecourseBuilder(solver="cplex", action_set=a, coefficients=coefficients, x=x)

    ## max items is the
    assert t_cplex.max_items == 3
    ## min items is the
    assert t_cplex.min_items == 0
    ## cost_type tells the MIP to solve for either the max objective function
    ## or the total cost
    assert t_cplex.mip_cost_type == 'max'

    ## solve mip.
    soln = t_cplex.fit()
    ## check cost
    assert np.isclose(soln['cost'], .5)
    ## check solution
    assert soln['actions'].tolist() == [1, 0, 0]
