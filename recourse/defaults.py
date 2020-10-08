# This file contains constants used in actionable-recourse

### Solver ###

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


_SOLVER_TYPE_CPX = 'cplex'
_SOLVER_TYPE_PYTHON_MIP = 'python-mip'

# Set Default Solver
def set_default_solver():

    if _check_solver_cpx():
        return _SOLVER_TYPE_CPX

    if _check_solver_python_mip():
        return _SOLVER_TYPE_PYTHON_MIP

    raise ModuleNotFoundError('could not find installed MIP solver')

DEFAULT_SOLVER = set_default_solver()

# Build List of Supported Solvers
SUPPORTED_SOLVERS = []

if _check_solver_cpx():
    SUPPORTED_SOLVERS.append(_SOLVER_TYPE_CPX)

if _check_solver_python_mip():
    SUPPORTED_SOLVERS.append(_SOLVER_TYPE_PYTHON_MIP)

SUPPORTED_SOLVERS = tuple(SUPPORTED_SOLVERS)


### Cost Function Types ###

VALID_MIP_COST_TYPES = {'total', 'local', 'max'}
DEFAULT_AUDIT_COST_TYPE = 'max'
DEFAULT_FLIPSET_COST_TYPE = 'local'

### Enumeration Types ###

VALID_ENUMERATION_TYPES = {'mutually_exclusive', 'distinct_subsets'}
DEFAULT_ENUMERATION_TYPE = 'distinct_subsets'
