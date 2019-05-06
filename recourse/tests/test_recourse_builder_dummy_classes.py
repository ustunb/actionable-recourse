from recourse.builder import RecourseBuilder
import numpy as np
from recourse.tests.test_classes import *

def test_actionset_fake_data(my_actionset_fake):
    ## check that the column-names are being stored correctly
    assert my_actionset_fake._names == ['a', 'b', 'c']

    ## check that the action set is not aligned
    assert my_actionset_fake.aligned == False

    ## test alignment
    coefficients = [1, 2, 3]
    my_actionset_fake.align(coefficients)
    assert my_actionset_fake.aligned == True

    ## test action grid
    x = [1, 1, 1]
    actions, percentiles = my_actionset_fake.feasible_grid(x=x)
    assert actions['a'].tolist() == [0, 1]
    assert actions['b'].tolist() == [0, 1, 2]
    assert actions['c'].tolist() == [0, 2, 3]

    assert np.isclose(percentiles['a'], [.5, 1], atol=1e-6).all()
    assert np.isclose(percentiles['b'], [0, .5, 1], atol=1e-6).all()
    assert np.isclose(percentiles['c'], [0, .5, 1], atol=1e-6).all()


def test_recourse_builder_cplex_fake_data(my_recourse_builder_cplex_fake):
    """Base test. Tests that the factory object instantiates a
    _RecourseCPLEX module and that it has correct initialization attributes."""

    ## max items is the
    assert my_recourse_builder_cplex_fake.max_items == 3
    ## min items is the
    assert my_recourse_builder_cplex_fake.min_items == 0
    ## cost_type tells the MIP to solve for either the max objective function
    ## or the total cost
    assert my_recourse_builder_cplex_fake.mip_cost_type == 'max'

    ## solve mip.
    soln = my_recourse_builder_cplex_fake.fit()
    ## check cost
    assert np.isclose(soln['cost'], .5)
    ## check solution
    assert soln['actions'].tolist() == [1, 0, 0]
