
import numpy as np
import pytest
from recourse.paths import *
import pyomo.environ
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder, _SOLVER_TYPE_CBC, _SOLVER_TYPE_CPX
from recourse.flipset import Flipset
from recourse.auditor import RecourseAuditor

@pytest.fixture(params = ['german'])
def data(request):

    data_name = request.param
    df = pd.read_csv(test_dir / ('%s_processed.csv' % data_name))
    df_headers = df.columns.to_list()

    outcome_name = df_headers[0]
    df_headers.remove(outcome_name)
    to_drop = [outcome_name]

    if data_name == 'german':
        to_drop.extend(['Gender', 'PurposeOfLoan', 'OtherLoansAtStore'])

    variable_names = [n for n in df_headers if n not in to_drop]

    data = {
        'data_name': data_name,
        'outcome_name': outcome_name,
        'variable_names': variable_names,
        'Y': df[outcome_name],
        'X': df[variable_names],
        }

    return data


@pytest.fixture(params = ['logreg'])
def classifier(request, data):
    if request.param == 'logreg':
        clf = LogisticRegression(max_iter = 1000, solver = 'lbfgs')
        clf.fit(data['X'], data['Y'])

    return clf


@pytest.fixture(params = ['mutable', 'immutable'])
def action_set(request, data):
    """Generate an action_set for German data."""
    # setup action_set

    action_set = ActionSet(X = data['X'])
    if request.param == 'immutable' and data['data_name'] == 'german':
        immutable_attributes = ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse']
        action_set[immutable_attributes].mutable = False
        action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
        action_set['CheckingAccountBalance_geq_0'].step_direction = 1

    return action_set


@pytest.fixture(params = [_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC])
def recourse_builder(request, classifier, action_set):
    action_set.align(classifier)
    rb = RecourseBuilder(solver = request.param,
                         action_set = action_set,
                         clf = classifier)

    return rb


@pytest.fixture(params = ['_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC'])
def auditor(request, classifier, action_set):
    return RecourseAuditor(clf = classifier, action_set = action_set, solver= request.param)


@pytest.fixture(params = ['_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC'])
def flipset(request, classifier, action_set):
    return Flipset(clf = classifier, action_set = action_set, solver= request.param)


@pytest.fixture
def recourse_builder_cpx(classifier, action_set):
    action_set.align(classifier)
    rb = RecourseBuilder(solver = _SOLVER_TYPE_CPX,
                         action_set = action_set,
                         clf = classifier)

    return rb


@pytest.fixture
def recourse_builder_cbc(classifier, action_set):
    action_set.align(classifier)
    rb = RecourseBuilder(solver = _SOLVER_TYPE_CBC,
                         action_set = action_set,
                         clf = classifier)

    return rb
