from recourse.tests.fixtures import *

# Test Strategy
# --------------------------------------------------------
# cost function:        local
# variable types:       all binary, mix
# # of variables in w:  1, >1
# recourse:             exists, does not exist
# action_set:           all actionable, all conditionally actionable, all immutable, mix

# fit
# populate

n = 50
def test_auditor(auditor, data, scores, threshold):
    """Test if the CPLEX auditor runs."""
    df = auditor.audit(X = data['X'].iloc[:n])