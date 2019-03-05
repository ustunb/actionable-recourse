from recourse.auditor import RecourseAuditor
import pytest


@pytest.fixture
def german_auditor_cplex(german_clf, german_actionset_aligned):
    return RecourseAuditor(clf=german_clf, action_set=german_actionset_aligned, solver='cplex')


@pytest.fixture
def german_auditor_pyomo(german_clf, german_actionset_aligned):
    return RecourseAuditor(clf=german_clf, action_set=german_actionset_aligned, solver='pyomo')


def test_auditor_cplex(german_auditor_cplex, german_X_test):
    """Test if the CPLEX auditor runs."""
    df_cpx = german_auditor_cplex.audit(X=german_X_test)

def test_auditor_pyomo(german_auditor_pyomo, german_X_test):
    """Test if the PYOMO auditor runs."""
    df_cbc = german_auditor_pyomo.audit(X = german_X_test)


def test_compare_auditors(german_auditor_cplex, german_auditor_pyomo, german_X_test):
    """Compare the outputs of the CPLEX and PYOMO auditors."""
    df_cpx = german_auditor_cplex.audit(X=german_X_test)
    df_cbc = german_auditor_pyomo.audit(X=german_X_test)

    ## TODO compare assert