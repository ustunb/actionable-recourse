from recourse.tests.fixtures import *
from recourse.action_set import ActionSet

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
    a.set_alignment(coefficients)
    assert a.alignment_known == True
    assert not np.isnan(a.flip_direction).any()
    assert not np.isnan(a.compatible).any()

    # changing y_desired changes the flip direction
    fd = np.array(a.flip_direction)
    a.y_desired = -a.y_desired
    assert np.all(fd == -np.array(a.flip_direction))

    # flipping coefficients changes the flip direction
    b = ActionSet(X = data['X'])
    b.set_alignment(-coefficients)
    assert np.all(fd == -np.array(b.flip_direction))


def test_subset_constraints(data):

    n_categorical = data['X_cat'].shape[1]
    if n_categorical > 1:

        print(n_categorical)
        X_cat = pd.get_dummies(data['X_cat'], prefix_sep = '_is_')
        names_cat = X_cat.columns.to_list()

        X = data['X'].join(X_cat)
        a = ActionSet(X, y_desired = 1)

        # add constraint
        assert len(a.constraints) == 0
        id = a.add_constraint(constraint_type = 'subset_limit', names = names_cat, lb = 1, ub = 1)
        assert len(a.constraints) == 1

        # remove constraint
        a.remove_constraint(id)
        assert len(a.constraints) == 0

        # add progressively larger constriants
        k = len(names_cat)
        for n in range(1, k):
            a.add_constraint(constraint_type = 'subset_limit', names = names_cat, lb = 0, ub = k)

        assert len(a.constraints) == k

    return True

