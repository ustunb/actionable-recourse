from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder
import numpy as np
import pytest
from recourse.paths import *
import pyomo.environ
import pandas as pd
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def my_actionset_fake():
    """Action set with fake values."""
    ## default values for "X" (dataset) and "names" (column names).
    X = np.array([[1, 2, 3], [2, 3, 4]])
    names = ['a', 'b', 'c']

    ## initialize.
    return ActionSet(X, names=names)


@pytest.fixture
def my_recourse_builder_cplex_fake(my_actionset_fake):
    """CPLEX Recourse builder with fake values."""
    x = [1, 1, 1]
    coefficients = [1, 2, 3]
    return RecourseBuilder(solver="cplex", action_set=my_actionset_fake, coefficients=coefficients, x=x)


@pytest.fixture
def german_data():
    """Load german credit-reporting data."""
    data_name = 'german'
    data_file = test_dir / ('%s_processed.csv' % data_name)

    ## load dataset
    data_df = pd.read_csv(data_file)
    return data_df


@pytest.fixture
def german_X(german_data):
    """Process German to remove categorical columns."""
    outcome_name = german_data.columns[0]
    return german_data.drop([outcome_name, 'Gender', 'PurposeOfLoan', 'OtherLoansAtStore'], axis=1)


@pytest.fixture
def german_y(german_data):
    """Return the response column for the German dataset."""
    outcome_name = german_data.columns[0]
    return german_data[outcome_name]


@pytest.fixture
def german_actionset_unaligned(german_X):
    """Generate an actionset for German data."""
    # setup actionset
    action_set = ActionSet(X = german_X)
    immutable_attributes = ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse']
    action_set[immutable_attributes].mutable = False
    action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
    action_set['CheckingAccountBalance_geq_0'].step_direction = 1
    return action_set


@pytest.fixture
def german_clf(german_X, german_y):
    # fit classifier
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(german_X, german_y)
    return clf


@pytest.fixture
def german_coef(german_clf):
    return german_clf.coef_[0]


@pytest.fixture
def german_intercept(german_clf):
    return german_clf.intercept_[0]


@pytest.fixture
def german_actionset_aligned(german_actionset_unaligned, german_coef):
    german_actionset_unaligned.align(coefficients=german_coef)
    return german_actionset_unaligned


@pytest.fixture
def german_scores(german_clf, german_X):
    return pd.Series(german_clf.predict_proba(german_X)[:, 1])


@pytest.fixture
def german_denied_individual(german_scores, german_X):
    p = german_scores.median()
    denied_individuals = german_scores.loc[lambda s: s <= p].index
    idx = denied_individuals[5]
    return german_X.iloc[idx].values


@pytest.fixture
def german_p(german_scores):
    return german_scores.median() ## or you can set a score-threshold, like .8


@pytest.fixture
def german_X_test(german_clf, german_X):
    # create sample points for test
    n_test_points = 10
    test_seed = 2338
    denied_idx = german_clf.predict(german_X) < 0
    return german_X.iloc[denied_idx].sample(n=n_test_points, random_state=test_seed)
