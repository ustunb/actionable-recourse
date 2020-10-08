# This file contains constants used in actionable-recourse

### Solver ###

_SOLVER_TYPE_CPX = 'cplex'
_SOLVER_TYPE_PYTHON_MIP = 'python-mip'
SUPPORTED_SOLVERS = {_SOLVER_TYPE_CPX, _SOLVER_TYPE_PYTHON_MIP}

def set_default_solver():

    if _check_solver_cpx():
        return _SOLVER_TYPE_CPX

    if _check_solver_python_mip():
        return _SOLVER_TYPE_PYTHON_MIP

    raise ModuleNotFoundError('could not find installed MIP solver')


def _check_solver_cpx():
    """
    :return: True if CPLEX if installed
    """
    chk = False
    try:
        import cplex
        chk = True
    except ImportError:
        pass

    return chk


def _check_solver_python_mip():
    chk = False
    try:
        import mip
        #todo add some check to make sure CBC is installed
        chk = True
    except ImportError:
        pass
    return chk


DEFAULT_SOLVER = set_default_solver()


### Cost Types ###

VALID_MIP_COST_TYPES = {'total', 'local', 'max'}
DEFAULT_AUDIT_COST_TYPE = 'max'
DEFAULT_FLIPSET_COST_TYPE = 'local'

### Enumeration Types ###

VALID_ENUMERATION_TYPES = {'mutually_exclusive', 'distinct_subsets'}
DEFAULT_ENUMERATION_TYPE = 'distinct_subsets'
