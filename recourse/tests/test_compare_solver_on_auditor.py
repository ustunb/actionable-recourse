from recourse.auditor import RecourseAuditor
import pytest
from recourse.tests.fixtures import *
from recourse.tests.test_classes import *
from recourse.builder import _SOLVER_TYPE_CBC, _SOLVER_TYPE_CPX

n = 50
@pytest.mark.parametrize('auditor', [_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC], indirect=True)
def test_auditor(auditor, data, scores, threshold):
    """Test if the CPLEX auditor runs."""
    df = auditor.audit(X=data['X'].iloc[:n])


def test_compare_auditors(auditor_cpx, auditor_cbc, data):
    """Compare the outputs of the CPLEX and PYOMO auditors."""
    df_cpx = auditor_cpx.audit(X=data['X'].iloc[:n])
    df_cbc = auditor_cbc.audit(X=data['X'].iloc[:n])
    print(df_cpx)
    print(df_cbc)

    assert np.isclose(df_cpx['cost'].fillna(0), df_cbc['cost'].fillna(0), atol=1e-4).all()
    assert all(df_cpx['feasible'].fillna(True) == df_cbc['feasible'].fillna(True))