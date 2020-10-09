# Test Strategy
# --------------------------------------------------------
# cost function:        local, max, total
# variable types:       all binary, mix
# # of variables in w:  1, >1
# recourse:             exists, does not exist
# action_set:           all compatible, all conditionally compatible, all immutable, mix

# fit
# populate
import pytest
import numpy as np
from recourse.defaults import SUPPORTED_SOLVERS
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder

def test_rb_fit_without_initialization(data, recourse_builder):
    """Test fitting on a denied individual, CPLEX."""

    # pick a denied individual
    try:
        output = recourse_builder.fit()
    except AssertionError:
        assert True
    else:
        assert False

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_rb_onehot_encoding(data, solver):

    if len(data['categorical_names']) == 1:

        # pick only the indicator variables
        names = data['onehot_names']
        k = len(names)
        X = data['X'][names]
        assert np.all(X.sum(axis = 1) == 1)

        #setup classifier of the form
        #w = [3, -1, -1, -1,...]
        #t = -1
        # score(x[0] = 1) =  3 -> yhat = +1
        # score(x[j] = 1) = -2 -> yhat = -1 for j = 1,2,...,k
        coefs = -np.ones(k)
        coefs[0] = 3.0
        intercept = -1.0

        # setup action set
        a = ActionSet(X)
        a.add_constraint('subset_limit', names = names, lb = 0, ub = 1)
        a.set_alignment(coefficients = coefs, intercept = intercept)
        rb = RecourseBuilder(action_set = a, coefficients = coefs, intercept = intercept, solver = solver)
        for j in range(1, k):

            x = np.zeros(k)
            x[j] = 1.0
            assert rb.score(x) < 0

            # set point
            rb.x = x

            # find optimal action
            info = rb.fit()
            a = info['actions']

            # validate solution
            x_new = x + a
            assert rb.score(x_new) > 0
            assert np.isclose(a[j], -1.0)
            assert np.isclose(np.sum(x_new), 1.0)


def test_rb_fit(data, recourse_builder, features):
    print(len(features))
    print(recourse_builder.n_variables)
    recourse_builder.x = features
    output = recourse_builder.fit()
    assert output['cost'] >= 0.0


def test_empty_fit(data, features, action_set, coefficients, classifier, recourse_builder):
    names = data['X'].columns.tolist()
    action_set.set_alignment(classifier)
    direction = np.sign(coefficients)
    ## force everything to be the opposite direction.
    for n, d in zip(names, direction):
        action_set[n].step_direction = -1 * d
    recourse_builder._action_set = action_set

    recourse_builder.x = features
    output = recourse_builder.fit()
    assert output['cost'] == np.inf
    assert output['feasible'] == False


def test_rb_ydesired(data, recourse_builder, classifier):
    assert True