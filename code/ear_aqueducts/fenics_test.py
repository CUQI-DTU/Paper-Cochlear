#%%
import dolfin as dl
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

class TimeDependantHeat:    
    def __init__(self, mesh, Vh_parameter, Vh_state, t_init, t_final, t_1, dt, u0, f, obs_locations=None, obs_times=None):
        
        bc_tol = 1E-14
        self.u0 = u0 # u0: Initial condition

        self.mesh = mesh
        self.Vh_parameter = Vh_parameter
        self.Vh_state = Vh_state
        self.t_init = t_init
        self.t_final = t_final
        self.t_1 = t_1 # observation starting time
        self.dt = dt
        self.sim_times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)
        self.obs_locations = obs_locations
        # expression 1 at the locations
        self.obs_point_src = dl.Function(Vh_state)
        for loc in self.obs_locations:
            p = dl.PointSource(Vh_state,dl.Point(loc,0),1.0)
            p.apply(self.obs_point_src.vector())

        self.obs_times = obs_times
            
        class LeftBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):  
                return on_boundary and abs(x[0]) < bc_tol

        Gamma_left = LeftBoundary()
       
        # Dirichlet BC for the forward and the adjoint operators
        u_exp  = dl.Expression("0.1", degree=1)
        self._state_bcs = dl.DirichletBC(Vh_state, u_exp, Gamma_left)
        p_exp  = dl.Constant(0.0)
        self._adjoint_bcs = dl.DirichletBC(Vh_state, p_exp, Gamma_left)

        v = dl.TestFunction(Vh_state)
        u = dl.TrialFunction(Vh_state)
        
        self.M_expr = u*v*dl.dx # mass matrix expression
        self._E_expr = None # Poisson operator
        self.f_expr = self.dt*dl.inner(f,v)*dl.dx # source term

        self.M = dl.assemble(self.M_expr)
        
        self.fwd_sol = TimeDependantSolution()
        self.adj_sol = TimeDependantSolution()


    def E_expr(self, k):
        """Returns the Poisson operator corresponding to k"""
        k_func = dl.Function(self.Vh_parameter, k)
        v = dl.TestFunction(self.Vh_state)
        u = dl.TrialFunction(self.Vh_state)

        self._E_expr = self.dt*dl.exp(k_func)*dl.inner(dl.nabla_grad(u),\
                dl.nabla_grad(v))*dl.dx

        return self._E_expr

    
    def solveFwd(self, x, tol=1e-9):
        self.fwd_sol.clear()
        uold = self.u0.vector().copy()
        self.fwd_sol.add(uold, self.t_init)
        
        u = dl.Vector()
        self.M.init_vector(u, 0)
        
        t = self.t_init
        v = dl.TestFunction(self.Vh_state)

        while t < self.t_final-0.5*self.dt:
            
            t += self.dt
            uold_func = dl.Function(self.Vh_state, uold)
            rhs_expr  = dl.inner(uold_func,v)*dl.dx
            A,b =dl.assemble_system(self.M_expr+self.E_expr(x), \
                 rhs_expr+self.f_expr, self._state_bcs)
            dl.solve(A,u,b)
            u_copy = dl.Vector()
            self.M.init_vector(u_copy,0)
            u_copy[:] = u[:]
            self.fwd_sol.add(u_copy, t)
            uold = u
    
    def solveAdj(self, x, rhs, tol=1e-9):
        self.adj_sol.clear()
        v = dl.TestFunction(self.Vh_state)
        
        # initialize pold, p, u and ud
        pold = dl.Vector()
        self.M.init_vector(pold,0)  
        pold.zero()

        p = dl.Vector()
        self.M.init_vector(p,0)

        u = dl.Vector()
        self.M.init_vector(u,0)

        ud = dl.Vector()
        self.M.init_vector(ud,0)

        t = self.t_final
        self.adj_sol.add(pold, t)
  



        while t > self.t_init- .5*self.dt:
            if np.any(np.isclose(t, self.obs_times)): 
                print('t: ', t)
                u = self.fwd_sol.solution_at_time(t)
                rhs_i = rhs.solution_at_time(t)

                obs_rhs =  dl.Function(self.Vh_state, rhs_i)
                obs_rhs = dl.project(obs_rhs*self.obs_point_src, self.Vh_state)
            else:
                obs_rhs = dl.Constant(0.0)
            pold_func = dl.Function(self.Vh_state, pold)
            rhs_expr = (pold_func*v+obs_rhs*v)*dl.dx  
            A,b =dl.assemble_system(self.M_expr+self.E_expr(x), \
                 rhs_expr, self._adjoint_bcs)
            dl.solve(A,p,b)
            pold = p
            t -= self.dt # Amal: I rearranged
            self.adj_sol.add(p.copy(), t)


    def evalGradientParameter(self,x):
  
        out = dl.Function(self.Vh_parameter).vector()
        out = self.applyCt(x)
        
        return out


    def applyCt(self, x):
        #dp adj
        #out
        out = dl.Function(self.Vh_state).vector()
        product = dl.Vector()
        self.M.init_vector(product, 0)
        out.zero()

        Ph = self.Vh_parameter
        Vh = self.Vh_state

        k_test = dl.TestFunction(Ph)
        v_trial = dl.TrialFunction(Ph)
        t = self.t_init
        k = x
        k_func = dl.Function(self.Vh_parameter, k)

        while t < self.t_final -0.5 *self.dt:
            t += self.dt

            u_func = [u[0] for u in self.fwd_sol if np.isclose(u[1], t)]
            assert len(u_func) == 1
            u_func = dl.Function(Vh_state, u_func[0])

            Ct_i = dl.assemble(self.dt*dl.exp(k_func)*k_test*\
                              dl.inner(dl.nabla_grad(u_func),\
                              dl.nabla_grad(v_trial))*dl.dx)

            dummy = dl.Vector()
            Ct_i.init_vector(dummy,0)
            self._adjoint_bcs.zero_columns(Ct_i, dummy)
        
            product.zero()
            dp_current = [dp[0] for dp in self.adj_sol if np.isclose(dp[1], t)]
            assert len(dp_current) == 1
            dp_current = dp_current[0]

            Ct_i.mult(dp_current, product)
            out.axpy(1.0, product)
        return out



    def plotTimes(self, times, qt):

        # Extract fwd solutions of which t is in times
        for t in times:
            sol_fun = dl.Function(self.Vh_state, qt.solution_at_time(t))
            p = dl.plot(sol_fun)




           #p.set_min_max(-1*MAX, 1*MAX)
class TimeDependantSolution:
    def __init__(self):
        self._sol = []
        self._times = []

    def add(self, sol, time):
        self._sol.append(sol)
        self._times.append(time)

    def __getitem__(self, key):
        return self._sol[key], self._times[key]
    
    def __len__(self):
        return len(self._sol)
    
    def __iter__(self):
        return iter(zip(self._sol, self._times))
    
    def solution_at_time(self, time, atol=None):
        if atol is None:
            # tol is 10% of the smallest time step
            atol = np.abs(0.1*np.min(np.diff(self._times)))        

        sol_list = [sol for sol, t in self if np.isclose(t, time, atol=atol)]
        assert len(sol_list) <= 1

        return sol_list[0] if len(sol_list) == 1 else None

    
    def clear(self):
        self._sol = []
        self._times = []

    def copy(self):
        out = TimeDependantSolution()
        for sol, time in self:
            out.add(sol.copy(), time)
        return out

       
if __name__ == "__main__":
    # CREATE 1D mesh
    mesh = dl.IntervalMesh(100, 0, 1.0)
    # CREATE function space for the parameter
    Vh_parameter = dl.FunctionSpace(mesh, 'CG', 1)
    # CREATE function space for the state
    Vh_state = dl.FunctionSpace(mesh, 'CG', 1)

    # Start and final time
    t_init = 0
    t_final = 0.01
    dt = 0.0001
    # Observation starting time
    t_1 = 0.00

    # Initial condition
    u0 = dl.Expression('0', degree=1)
    u0 = dl.interpolate(u0, Vh_state)

    # Source term
    f = dl.Expression('0', degree=1)
    f = dl.interpolate(f, Vh_state)

    # data

    # CREATE TimeDependantHeat object
    locations = [0.1, 0.2, 0.3, 0.4, 0.5]
    fwd = TimeDependantHeat(mesh, Vh_parameter, Vh_state, t_init, t_final, t_1, dt, u0, f, locations)
    fwd.obs_times = fwd.sim_times[[10, 20, 30, 40, 50]]

    # Create parameter vector as a sine function
    #k = dl.Expression('sin(2*pi*x[0])+1.1', degree=1)
    k = dl.Expression('std::log(1.1 +sin(x[0]*2*pi))', degree=1)
    k_fun = dl.interpolate(k, Vh_parameter)
    k_vec = k_fun.vector()

    # Solve the fwd problem
    # time the fwd solve
    start = time.time()
    fwd.solveFwd(k_vec)
    end = time.time()
    print('fwd solve time: ', end-start)

    fwd.plotTimes(fwd.sim_times, fwd.fwd_sol)
    plt.ylim(-0.1, 0.15)
    plt.title('Forward solution, for exact parameter')

    data_exact = fwd.fwd_sol.copy()
    fwd.fwd_sol.clear()

    k_const = dl.Expression('0', degree=1)
    k_const = dl.interpolate(k_const, Vh_parameter).vector()

    # Solve the fwd problem
    fwd.solveFwd(k_const)
    plt.figure()
    fwd.plotTimes(fwd.sim_times, fwd.fwd_sol)
    plt.ylim(-0.1, 0.15)
    plt.title('Forward solution, for constant parameter')

    # Solve the adj problem
    def compute_rhs(fwd):
        rhs = TimeDependantSolution()
        for i in range(len(fwd.fwd_sol)):
            rhs_i = fwd.fwd_sol[i][0].copy()
            rhs_i.axpy(-1, data_exact[i][0])
            rhs_i *= -dt
            rhs.add(rhs_i, fwd.fwd_sol[i][1])
        return rhs

    # plot rhs
    plt.figure()
    rhs = compute_rhs(fwd)
    fwd.plotTimes(fwd.sim_times, rhs)
    plt.title('rhs')
    
    # time the adj solve
    start = time.time()
    fwd.solveAdj(k_const, rhs)
    end = time.time()
    print('adj solve time: ', end-start)

    # Plot the adjoint solution
    plt.figure()
    fwd.plotTimes(fwd.sim_times, fwd.adj_sol)


    # Compute the gradient
    grad = fwd.evalGradientParameter(k_const)


    # cost function
    def cost(k):
        # k from numpy array to dolfin vector
        k_v = dl.Function(Vh_parameter).vector()
        k_v[:] = k[:]
        fwd.solveFwd(k_v)
        cost = 0
        for i in range(len(fwd.fwd_sol)):
            u_i = fwd.fwd_sol[i][0]
            data_i = data_exact[i][0]
            diff_i = u_i.copy()
            diff_i.axpy(-1, data_i)
            diff_i_func = dl.Function(Vh_state, diff_i)
            diff_i_func_pts = dl.project(fwd.obs_point_src*diff_i_func, Vh_state)
            diff_i = diff_i_func_pts.vector()
            if np.any(np.isclose(fwd.sim_times[i], fwd.obs_times)):
                #print('t: ', fwd.sim_times[i])
                cost += dt*0.5*diff_i.inner(fwd.M*diff_i)
        return cost
    verify_grad_const = True
    if verify_grad_const:
        # gradient check using scipy.optimize.approx_fprime
        from scipy.optimize import approx_fprime
        grad_scipy = approx_fprime(k_const.get_local(), cost, 1e-5)
        grad_scipy_const_func = dl.Function(Vh_parameter)
        grad_scipy_const_func.vector()[:] = grad_scipy[:]

        # Plot the gradients
        plt.figure()
        dl.plot(dl.Function(Vh_parameter, grad), label='adj')
        dl.plot(grad_scipy_const_func, label='scipy')
        plt.title('gradient at constant parameter')
        plt.legend()


        #
    verify_grad_true = False
    if verify_grad_true: 
        # compute gradient at true
        fwd.solveFwd(k_vec)
        rhs = compute_rhs(fwd)
        fwd.solveAdj(k_vec, rhs)
        grad_true = fwd.evalGradientParameter(k_vec)
        plt.figure()
        dl.plot(dl.Function(Vh_parameter, grad_true))
        plt.title('gradient at true')
        #
        grad_scipy_true = approx_fprime(k_vec.get_local(), cost, 1e-8)
        plt.figure()
        plt.plot(grad_scipy_true[::-1]) 
        plt.title('scipy gradient at true')

    verify_grad_random = False
    if verify_grad_random: 
        random_k = dl.Vector()
        fwd.M.init_vector(random_k, 0)
        random_k.set_local(np.random.rand(random_k.local_size()))
        # compute gradient at random
        fwd.solveFwd(random_k)
        rhs = compute_rhs(fwd)
        fwd.solveAdj(random_k, rhs)
        grad_random = fwd.evalGradientParameter(random_k)
        plt.figure()
        dl.plot(dl.Function(Vh_parameter, grad_random))
        plt.title('gradient at random')
        
        grad_scipy_random = approx_fprime(random_k.get_local(), cost, 1e-8)
        plt.figure()
        plt.plot(grad_scipy_random[::-1])
        plt.title('scipy gradient at random')



# %%
