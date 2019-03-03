from recourse.builder import RecourseBuilder
from recourse.action_set import ActionSet
import numpy as np

class MyActionSet(ActionSet):
    def __init__(self):
        ## default values for "X" (dataset) and "names" (column names).
        X = np.array([[1, 2, 3], [2, 3, 4]])
        names = ['a', 'b', 'c']

        ## initialize.
        super().__init__(X, names=names)

        ## align coefficients.
        coefficients = [1, 2, 3]
        self.align(coefficients)


def test_recourse_builder_cplex():
    """Base test. Tests that the factory object instantiates a
    _RecourseCPLEX module and that it has correct initialization attributes."""

    a = MyActionSet()
    x = [1, 1, 1]

    t_cplex = RecourseBuilder(solver="cplex", action_set=a)