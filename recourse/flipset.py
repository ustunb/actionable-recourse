from recourse._flipset_base import _FlipsetBase
from recourse._flipset_cplex import _FlipsetBuilderCPLEX
from recourse._flipset_pyomo import _FlipsetBuilderPyomo

class FlipsetBuilder(_FlipsetBase):
    """Factory Method."""
    def __new__(cls, action_set, coefficients, intercept = 0.0, x = None, optimizer="cplex", **kwargs):
        if optimizer == "cplex":
            return (super()
                    .__new__(_FlipsetBuilderCPLEX)
                    .__init__(action_set, coefficients, intercept = intercept, x = x, **kwargs)
                    )
        elif optimizer == "cbc":
            return (super()
                    .__new__(_FlipsetBuilderPyomo)
                    .__init__(action_set, coefficients, intercept = intercept, x = x, **kwargs))
        else:
            raise NameError("pick optimizer in: ['cplex', 'cbc']")