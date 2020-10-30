from recourse import ActionSet
import numpy as np

# Test Strategy
# --------------------------------------------------------
# variable types:       all, binary, mix
# action_set:           all compatible, all conditionally compatible, all immutable, mix


def test_initialization(data):

    a = ActionSet(X = data['X'])
    b = ActionSet(X = data['X'].values, names = data['X'].columns.tolist())
    assert a.name == b.name


def test_y_desired(data):

    # initialization checks
    a = ActionSet(data['X'], y_desired = 1)
    assert a.y_desired == 1

    a = ActionSet(data['X'], y_desired = -1)
    assert a.y_desired == -1

    a = ActionSet(data['X'], y_desired = 0)
    assert a.y_desired == -1

    # setter checks
    a.y_desired = 1.0
    assert a.y_desired == 1

    a.y_desired = -1.0
    assert a.y_desired == -1

    a.y_desired = 0.0
    assert a.y_desired == -1


def test_align(data, coefficients):

    a = ActionSet(X = data['X'])

    # no alignment means flip direction and compatability are empty
    assert a.alignment_known == False
    assert np.isnan(a.flip_direction).all()
    assert np.isnan(a.compatible).all()

    # aligning sets compatability and flip direction
    a._align(coefficients)
    assert a.alignment_known == True
    assert not np.isnan(a.flip_direction).any()
    assert not np.isnan(a.compatible).any()

    # changing y_desired changes the flip direction
    fd = np.array(a.flip_direction)
    a.y_desired = -a.y_desired
    assert np.all(fd == -np.array(a.flip_direction))

    # flipping coefficients changes the flip direction
    b = ActionSet(X = data['X'])
    b._align(-coefficients)
    assert np.all(fd == -np.array(b.flip_direction))


def test_subset_constraints(data):

    if len(data['categorical_names']) == 1:

        a = ActionSet(data['X'], y_desired = 1)

        # add constraint
        assert len(a.constraints) == 0
        id = a.add_constraint(constraint_type = 'subset_limit', names = data['onehot_names'], lb = 1, ub = 1)
        assert len(a.constraints) == 1

        # remove constraint
        a.remove_constraint(id)
        assert len(a.constraints) == 0

        # add progressively larger constriants
        k = len(data['onehot_names'])
        for n in range(k):
            a.add_constraint(constraint_type = 'subset_limit', names = data['onehot_names'], lb = 0, ub = n)

        assert len(a.constraints) == k

    return True

