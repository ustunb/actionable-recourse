import numpy as np

### Solver ###

_SOLVER_TYPE_CPX = 'cplex'
_SOLVER_TYPE_CBC = 'cbc'
SUPPORTED_SOLVERS = {_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC}

def set_default_solver():

    if _check_solver_cpx():
        return _SOLVER_TYPE_CPX

    if _check_solver_cbc():
        return _SOLVER_TYPE_CBC

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


def _check_solver_cbc():
    chk = False
    try:
        import pyomo
        #todo add some check to make sure CBC is installed
        chk = True
    except ImportError:
        pass
    return chk


DEFAULT_SOLVER = set_default_solver()
