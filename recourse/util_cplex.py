# This file contains convenience functions for CPLEX MIP Objects
import numpy as np
from functools import reduce
from cplex import Cplex, SparsePair
from cplex.exceptions import CplexError

#Copying
def copy_cplex(cpx):
    """
    Copy a Cplex object
    :param cpx: Cplex object
    :return: Copy of Cplex object
    """
    cpx_copy = Cplex(cpx)
    cpx_parameters = cpx.parameters.get_changed()
    for (pname, pvalue) in cpx_parameters:
        phandle = reduce(getattr, str(pname).split("."), cpx_copy)
        phandle.set(pvalue)
    return cpx_copy


# Building
def add_variable_cpx(cpx, name, obj, ub, lb, vtype):
    """
    Convenience function to add a variable to a Cplex() object
    :param cpx: handle to Cplex() object
    :param name: variable name
    :param obj: coefficient in linear objective
    :param ub: upper bound on variable
    :param lb: lower bound on variable
    :param vtype: variable type
    :return:
    """

    # name
    if isinstance(name, np.ndarray):
        name = name.tolist()
    elif isinstance(name, str):
        name = [name]

    nvars = len(name)

    # convert inputs
    if nvars == 1:

        # convert to list
        name = name if isinstance(name, list) else [name]
        obj = [float(obj[0])] if isinstance(obj, list) else [float(obj)]
        ub = [float(ub[0])] if isinstance(ub, list) else [float(ub)]
        lb = [float(lb[0])] if isinstance(lb, list) else [float(lb)]
        vtype = vtype if isinstance(vtype, list) else [vtype]

    else:

        # convert to list
        if isinstance(vtype, np.ndarray):
            vtype = vtype.tolist()
        elif isinstance(vtype, str):
            if len(vtype) == 1:
                vtype = nvars * [vtype]
            elif len(vtype) == nvars:
                vtype = list(vtype)
            else:
                raise ValueError('invalid length: len(vtype) = %d. expected either 1 or %d' % (len(vtype), nvars))

        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        elif isinstance(obj, list):
            if len(obj) == nvars:
                obj = [float(v) for v in obj]
            elif len(obj) == 1:
                obj = nvars * [float(obj)]
            else:
                raise ValueError('invalid length: len(obj) = %d. expected either 1 or %d' % (len(obj), nvars))
        else:
            obj = nvars * [float(obj)]

        if isinstance(ub, np.ndarray):
            ub = ub.tolist()
        elif isinstance(ub, list):
            if len(ub) == nvars:
                ub = [float(v) for v in ub]
            elif len(ub) == 1:
                ub = nvars * [float(ub)]
            else:
                raise ValueError('invalid length: len(ub) = %d. expected either 1 or %d' % (len(ub), nvars))
        else:
            ub = nvars * [float(ub)]

        if isinstance(lb, np.ndarray):
            lb = lb.tolist()
        elif isinstance(lb, list):
            if len(lb) == nvars:
                lb = [float(v) for v in lb]
            elif len(ub) == 1:
                lb = nvars * [float(lb)]
            else:
                raise ValueError('invalid length: len(lb) = %d. expected either 1 or %d' % (len(lb), nvars))
        else:
            lb = nvars * [float(lb)]

    # check that all components are lists
    assert isinstance(name, list)
    assert isinstance(obj, list)
    assert isinstance(ub, list)
    assert isinstance(lb, list)
    assert isinstance(vtype, list)

    # check components
    for n in range(nvars):
        assert isinstance(name[n], str)
        assert isinstance(obj[n], float)
        assert isinstance(ub[n], float)
        assert isinstance(lb[n], float)
        assert isinstance(vtype[n], str)

    if (vtype.count(vtype[0]) == len(vtype)) and vtype[0] == cpx.variables.type.binary:
        cpx.variables.add(names = name, obj = obj, types = vtype)
    else:
        cpx.variables.add(names = name, obj = obj, ub = ub, lb = lb, types = vtype)


# Parameter Setting
DEFAULT_CPLEX_PARAMETERS = {
    #
    'display_cplex_progress': False,
    #set to True to show CPLEX progress in console
    #
    'n_cores': 1,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain control callbacks in CPLEX 12.7.0 and earlier
    #
    'randomseed': 0,
    # This parameter sets the random seed differently for diversity of solutions.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    #
    'time_limit': 1e75,
    # runtime before stopping,
    #
    'node_limit': 9223372036800000000,
    # number of nodes to process before stopping,
    #
    'mipgap': np.finfo('float').eps,
    # Sets a relative tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html
    #
    'absmipgap': np.finfo('float').eps,
    # Sets an absolute tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # When this difference falls below the value of this parameter, the mixed integer optimization is stopped.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpAGap.html
    #
    'objdifference': 0.0,
    # Used to update the cutoff each time a mixed integer solution is found. This value is subtracted from objective
    # value of the incumbent update, so that the solver ignore solutions that will not improve the incumbent by at
    # least this amount.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ObjDif.html#
    #
    'integrality_tolerance': 0.0,
    # specifies the amount by which an variable can differ from an integer and be considered integer feasible. 0 is OK
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpInt.html
    #
    'mipemphasis': 0,
    # Controls trade-offs between speed, feasibility, optimality, and moving bounds in MIP.
    # 0     =	Balance optimality and feasibility; default
    # 1	    =	Emphasize feasibility over optimality
    # 2	    =	Emphasize optimality over feasibility
    # 3 	=	Emphasize moving best bound
    # 4	    =	Emphasize finding hidden feasible solutions
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIPEmphasis.html
    #
    'bound_strengthening': -1,
    # Decides whether to apply bound strengthening in mixed integer programs (MIPs).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BndStrenInd.html
    # -1    = cplex chooses
    # 0     = no bound strengthening
    # 1     = bound strengthening
    #
    'cover_cuts': -1,
    # Decides whether or not cover cuts should be generated for the problem.
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Covers.html
    # -1    = Do not generate cover cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate cover cuts moderately
    # 2	    = Generate cover cuts aggressively
    # 3     = Generate cover cuts very  aggressively
    #
    'zero_half_cuts': -1,
    # Decides whether or not to generate zero-half cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ZeroHalfCuts.html
    # -1    = Do not generate MIR cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate MIR cuts moderately
    # 2	    = Generate MIR cuts aggressively
    #
    'mir_cuts': -1,
    # Decides whether or not to generate mixed-integer rounding cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIRCuts.html
    # -1    = Do not generate zero-half cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate zero-half cuts moderately
    # 2	    = Generate zero-half cuts aggressively
    #
    'implied_bound_cuts': 0,
    # Decides whether or not to generate valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    'locally_implied_bound_cuts': 3,
    # Decides whether or not to generate locally valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    'scale_parameters': 0,
    # Decides how to scale the problem matrix.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ScaInd.html
    # 0     = equilibration scaling
    # 1     = aggressive scaling
    # -1    = no scaling
    #
    'numerical_emphasis': 0,
    # Emphasizes precision in numerically unstable or difficult problems.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/NumericalEmphasis.html
    # 0     = off
    # 1     = on
    #
    'poolsize': 100,
    # Limits the number of solutions kept in the solution pool
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolCapacity.html
    # number of feasible solutions to keep in solution pool
    #
    'poolrelgap': float('nan'),
    # Sets a relative tolerance on the objective value for the solutions in the solution pool.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolGap.html
    #
    'poolreplace': 2,
    # Designates the strategy for replacing a solution in the solution pool when the solution pool has reached its capacity.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolReplace.html
    # 0	= Replace the first solution (oldest) by the most recent solution; first in, first out; default
    # 1	= Replace the solution which has the worst objective
    # 2	= Replace solutions in order to build a set of diverse solutions
    #
    'repairtries': 20,
    # Limits the attempts to repair an infeasible MIP start.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RepairTries.html
    # -1	None: do not try to repair
    #  0	Automatic: let CPLEX choose; default
    #  N	Number of attempts
    #
    'nodefilesize': (120 * 1024) / 1,
    # size of the node file (for large scale problems)
    # if the B & B can no longer fit in memory, then CPLEX stores the B & B in a node file
    }


def set_cpx_display_options(cpx, display_mip = True, display_parameters = False, display_lp = False):
    """
    Convenience function to turn on/off CPLEX functions
    :param cpx:
    :param display_mip:
    :param display_parameters:
    :param display_lp:
    :return:
    """
    cpx.parameters.mip.display.set(display_mip)
    cpx.parameters.simplex.display.set(display_lp)
    cpx.parameters.paramdisplay.set(display_parameters)

    if not (display_mip or display_lp):
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)

    return cpx


def set_cpx_parameters(cpx, param = DEFAULT_CPLEX_PARAMETERS):
    """
    Set parameters of a Cplex object
    :param cpx: Cplex object
    :param param: dictionary of parameters
    :return: cpx
    """

    # get parameter handle
    p = cpx.parameters

    # Record calls to C API
    # cpx.parameters.record.set(True)

    if param['display_cplex_progress'] is (None or False):
        cpx = set_cpx_display_options(cpx, display_mip = False, display_lp =  False, display_parameters = False)

    # major parameters
    p.randomseed.set(param['randomseed'])
    p.threads.set(param['n_cores'])
    p.output.clonelog.set(0)
    p.parallel.set(1)

    # solution strategy
    p.emphasis.mip.set(param['mipemphasis'])
    p.preprocessing.boundstrength.set(param['bound_strengthening'])

    # cuts
    p.mip.cuts.implied.set(param['implied_bound_cuts'])
    p.mip.cuts.localimplied.set(param['locally_implied_bound_cuts'])
    p.mip.cuts.zerohalfcut.set(param['zero_half_cuts'])
    p.mip.cuts.mircut.set(param['mir_cuts'])
    p.mip.cuts.covers.set(param['cover_cuts'])
    #
    # tolerances
    p.emphasis.numerical.set(param['numerical_emphasis'])
    p.mip.tolerances.integrality.set(param['integrality_tolerance'])

    # initialization
    p.mip.limits.repairtries.set(param['repairtries'])

    # solution pool
    p.mip.pool.capacity.set(param['poolsize'])
    p.mip.pool.replace.set(param['poolreplace'])

    # stopping
    p.mip.tolerances.mipgap.set(param['mipgap'])
    p.mip.tolerances.absmipgap.set(param['absmipgap'])

    if param['time_limit'] < DEFAULT_CPLEX_PARAMETERS['time_limit']:
        cpx = set_cpx_time_limit(cpx, param['time_limit'])

    if param['node_limit'] < DEFAULT_CPLEX_PARAMETERS['node_limit']:
        cpx = set_cpx_node_limit(cpx, param['node_limit'])

    return cpx


def get_cpx_parameters(cpx):
    """
    :param cpx: Cplex object
    :return: Cplex object
    """

    p = cpx.parameters

    param = {
        # major
        'display_cplex_progress': p.mip.display.get() > 0,
        'randomseed': p.randomseed.get(),
        'n_cores': p.threads.set.get(),
        #
        # strategy
        'mipemphasis': p.emphasis.mip.get(),
        'scale_parameters': p.read.scale.get(),
        'locally_implied_bound_cuts': p.mip.cuts.localimplied.get(),
        #
        # stopping
        'time_limit': p.mip.timelimit.get(),
        'node_limit': p.mip.limits.nodes.get(),
        'mipgap': p.mip.tolerances.mipgap.get(),
        'absmipgap': p.mip.tolerances.absmipgap.get(),
        #
        # mip tolerances
        'integrality_tolerance':p.mip.tolerances.integrality.get(),
        'numerical_emphasis': p.parameters.emphasis.numerical.get(),
        #
        # solution pool
        'repairtries': p.mip.limits.repairtries.get(),
        'poolsize': p.mip.pool.capacity.get(),
        'poolreplace': p.mip.pool.replace.get(),
        #
        # node file
        # mip.parameters.workdir.Cur  = exp_workdir;
        # mip.parameters.workmem.Cur                    = cplex_workingmem;
        # mip.parameters.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        # mip.parameters.mip.limits.treememory.Cur      = cplex_nodefilesize;
        }

    return param


def set_mip_cutoff_values(cpx, objval, objval_increment):
    """
    Set the cutoff values used by the Cplex solver
    :param cpx: Cplex object
    :param objval:
    :param objval_increment:
    :return:
    """
    assert objval >= 0.0
    assert objval_increment >= 0.0
    p = cpx.parameters
    p.mip.tolerances.uppercutoff.set(float(objval))
    p.mip.tolerances.objdifference.set(0.95 * float(objval_increment))
    p.mip.tolerances.absmipgap.set(0.95 * float(objval_increment))
    return cpx


def set_cpx_time_limit(cpx, time_limit = None):
    """
    Convenience function to set a time limit on a Cplex object
    :param cpx: Cplex object
    :param time_limit: time limit in seconds
    :return: cpx: Cplex object
    """
    max_time_limit = float(cpx.parameters.timelimit.max())

    if time_limit is None:
        time_limit = max_time_limit
    else:
        time_limit = float(time_limit)
        time_limit = min(time_limit, max_time_limit)

    assert time_limit >= 0.0
    cpx.parameters.timelimit.set(time_limit)
    return cpx


def set_cpx_node_limit(cpx, node_limit = None):
    """
    Convenience function to set a node limit on a Cplex object.
    The node limit determines the maximum number of nodes that can be solved in
    branch and bound.
    :param cpx: Cplex object
    :param node_limit: time limit in seconds
    :return: cpx: Cplex object
    """
    max_node_limit = cpx.parameters.mip.limits.nodes.max()
    if node_limit == float('inf'):
        node_limit = max_node_limit
    elif node_limit is None:
        node_limit = max_node_limit
    else:
        node_limit = int(node_limit)
        node_limit = min(node_limit, max_node_limit)
    assert node_limit >= 0.0
    cpx.parameters.mip.limits.nodes.set(node_limit)
    return cpx


def toggle_cpx_preprocessing(cpx, toggle = True):
    """
    Toggle preprocessing on a Cplex object.
    This function is helpful for debugging, and running computational experiments
    :param cpx: Cplex object
    :param toggle: set to True to turn on pre-processing
    :return: Cplex object
    """
    # presolve
    # mip.parameters.preprocessing.presolve.help()
    # 0 = off
    # 1 = on

    # boundstrength
    # type of bound strengthening  :
    # -1 = automatic
    # 0 = off
    # 1 = on

    # reduce
    # mip.parameters.preprocessing.reduce.help()
    # type of primal and dual reductions  :
    # 0 = no primal and dual reductions
    # 1 = only primal reductions
    # 2 = only dual reductions
    # 3 = both primal and dual reductions

    # coeffreduce strength
    # level of coefficient reduction  :
    #   -1 = automatic
    #   0 = none
    #   1 = reduce only to integral coefficients
    #   2 = reduce any potential coefficient
    #   3 = aggressive reduction with tilting

    # dependency
    # indicator for preprocessing dependency checker  :
    #   -1 = automatic
    #   0 = off
    #   1 = at beginning
    #   2 = at end
    #   3 = at both beginning and end
    if toggle:
        cpx.parameters.preprocessing.aggregator.reset()
        cpx.parameters.preprocessing.reduce.reset()
        cpx.parameters.preprocessing.presolve.reset()
        cpx.parameters.preprocessing.coeffreduce.reset()
        cpx.parameters.preprocessing.boundstrength.reset()
    else:
        cpx.parameters.preprocessing.aggregator.set(0)
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.preprocessing.presolve.set(0)
        cpx.parameters.preprocessing.coeffreduce.set(0)
        cpx.parameters.preprocessing.boundstrength.set(0)

    return cpx


# Solution Statistics
def get_stats_cpx(cpx):
    """
    Returns information associated with the current best solution for the mip
    :param cpx: Cplex object
    :return: dictionary containing information about the solution to cpx
    """

    INITIAL_SOLUTION_INFO = {
        'status': 'no solution exists',
        'status_code': float('nan'),
        'has_solution': False,
        'has_mipstats': False,
        'iterations': 0,
        'nodes_processed': 0,
        'nodes_remaining': 0,
        'values': float('nan'),
        'objval': float('nan'),
        'upperbound': float('nan'),
        'lowerbound': float('nan'),
        'gap': float('nan'),
        }

    info = dict(INITIAL_SOLUTION_INFO)
    try:
        sol = cpx.solution
        progress_info = {
            'status': sol.get_status_string(),
            'status_code': sol.get_status(),
            'iterations': sol.progress.get_num_iterations(),
            'nodes_processed': sol.progress.get_num_nodes_processed(),
            'nodes_remaining': sol.progress.get_num_nodes_remaining()}
        info.update(progress_info)
        info['has_mipstats'] = True
    except CplexError:
        pass

    try:
        sol = cpx.solution
        solution_info = {
            'values': np.array(sol.get_values()),
            'objval': sol.get_objective_value(),
            'upperbound': sol.MIP.get_cutoff(),
            'lowerbound': sol.MIP.get_best_objective(),
            'gap': sol.MIP.get_mip_relative_gap()}
        info.update(solution_info)
        info['has_solution'] = True
    except CplexError:
        pass

    return info


# Initialization
def add_mip_start_cpx(cpx, solution, effort_level = 1, name = None):
    """
    :param cpx: Cplex object
    :param solution: solution vector (list or np.array)
    :param effort_level:    (must be one of the values of mip.MIP_starts.effort_level)
                            1 <-> check_feasibility
                            2 <-> solve_fixed
                            3 <-> solve_MIP
                            4 <-> repair
                            5 <-> no_check
    :param name: name to identify the solution
    :return: Cplex object
    """
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()

    mip_start = SparsePair(val = solution, ind = list(range(len(solution))))
    if name is None:
        cpx.MIP_starts.add(mip_start, effort_level)
    else:
        cpx.MIP_starts.add(mip_start, effort_level, name)

    return cpx
