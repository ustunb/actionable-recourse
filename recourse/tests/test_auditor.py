# Test Strategy
# --------------------------------------------------------
# cost function:        local
# variable types:       all binary, mix
# # of variables in w:  1, >1
# recourse:             exists, does not exist
# action_set:           all compatible, all conditionally compatible, all immutable, mix

# fit
# populate

n = 50
def test_auditor(auditor, data, scores, threshold):
    """test auditor"""
    df = auditor.audit(X = data['X'].iloc[:n])

    # todo: check that points that receive desired outcome = NaN
    assert len(df) == n

    # todo: check that points that receive desired outcome = NaN

    # todo: check that points without recourse have cost = Inf

    # todo: check that points have the same x-values have the same cost/feasibility

