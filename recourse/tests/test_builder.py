from recourse.tests.fixtures import *

# Test Strategy
# --------------------------------------------------------
# cost function:        local, max, total
# variable types:       all binary, mix
# # of variables in w:  1, >1
# recourse:             exists, does not exist
# action_set:           all actionable, all conditionally actionable, all immutable, mix

# fit
# populate

def test_rb_basic(data, recourse_builder):
    print(recourse_builder)


def test_rb_fit_without_initialization(data, recourse_builder):
    """Test fitting on a denied individual, CPLEX."""

    # pick a denied individual
    try:
        output = recourse_builder.fit()
    except AssertionError:
        assert True
    else:
        assert False


def test_rb_fit(data, recourse_builder, features):
    print(len(features))
    print(recourse_builder.n_variables)
    recourse_builder.x = features
    output = recourse_builder.fit()
    assert output['cost'] >= 0.0


def test_empty_fit(data, features, action_set, coefficients, classifier, recourse_builder):
    names = data['X'].columns.tolist()
    action_set.align(classifier)
    direction = np.sign(coefficients)
    ## force everything to be the opposite direction.
    for n, d in zip(names, direction):
        action_set[n].step_direction = -1 * d
    recourse_builder._action_set = action_set

    recourse_builder.x = features
    output = recourse_builder.fit()
    assert output['cost'] == np.inf
    assert output['feasible'] == False