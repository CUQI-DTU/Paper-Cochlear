import dolfin as dl
import numpy as np

# Expression for time dependent boundary condition
class TimeDependentBoundaryCondition(dl.UserExpression):
    def __init__(self, times, bc_data, **kwargs):
        self.t = 0
        self.times = times
        self.bc_data = bc_data
        super().__init__(**kwargs)
    def eval(self, value, x):
        value[0] = np.interp(self.t,
                             self.times,
                             self.bc_data)
    def value_shape(self):
        return ()