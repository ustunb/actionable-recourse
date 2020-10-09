import numpy as np
n = 50
#todo choose points randomly

def test_compare_flipsets(flipset_cpx, flipset_python_mip):
    cpx_items = flipset_cpx.populate(total_items=5).items
    cbc_items = flipset_python_mip.populate(total_items=5).items

    assert len(cpx_items) == len(cbc_items)
    assert np.isclose([x['cost'] for x in cpx_items], [x['cost'] for x in cbc_items]).all()


def test_compare_auditors(auditor_cpx, auditor_python_mip, data):
    """Compare the outputs of the CPLEX and Python MIP auditors."""
    df_cpx = auditor_cpx.audit(X=data['X'].iloc[:n])
    df_cbc = auditor_python_mip.audit(X=data['X'].iloc[:n])
    assert np.isclose(df_cpx['cost'].fillna(0), df_cbc['cost'].fillna(0), atol=1e-4).all()
    assert all(df_cpx['feasible'].fillna(True) == df_cbc['feasible'].fillna(True))