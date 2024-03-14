#%%
import dolfin as dl
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import scipy as sp
from scipy.interpolate import interp1d
#import cuqi
#import cuqipy_fenics

class TimeDependantHeat:    
    def __init__(self, L, nx, t_init, t_final, dt, u0, f, bc_exp, bc_domain, obs_locations=None, obs_times=None):
        self.L = L
        self.nx = nx
        self.mesh = dl.IntervalMesh(nx, 0, L)
        # add nodes close to loc

        level = 2
        print("####", obs_locations)
        for level_i in range(level):
            markers = dl.MeshFunction("bool", self.mesh, self.mesh.topology().dim(), False)
            for cell in dl.cells(self.mesh):
                for loc in obs_locations:
                    markers[cell] = markers[cell] or cell.contains(dl.Point(loc))
            print("#####sum", sum(markers.array()))

            self.mesh = dl.refine(self.mesh, markers)
        self.mesh.coordinates().sort(0)

        # create new mesh with sorted nodes
        mesh2 = dl.Mesh()
        editor = dl.MeshEditor()
        editor.open(mesh2, 'interval', 1, 1)
        editor.init_vertices(self.mesh.num_vertices())
        editor.init_cells(self.mesh.num_cells())

        for i, coor in enumerate(self.mesh.coordinates()):
            editor.add_vertex(i, coor)
            if i > 0:
                editor.add_cell(i-1, [i-1, i])
        editor.close()
        mesh2.init()
        self.mesh = mesh2

        #coor = fwd.mesh.coordinates().sort(0)

        self.Vh_parameter = dl.FunctionSpace(self.mesh, 'CG', 1)
        self.Vh_state = dl.FunctionSpace(self.mesh, 'CG', 1)
        
        # Initial condition
        self.u0 = dl.interpolate(u0, self.Vh_state)

        # Source term
        self.f = dl.interpolate(f, self.Vh_state)
        self.t_init = t_init
        self.t_final = t_final
        self.dt = dt
        self.sim_times = np.arange(self.t_init, self.t_final+.5*self.dt, self.dt)
        self.obs_locations = obs_locations

        self.obs_times = obs_times
        self.f_dirac = self.mesh.hmin()*1.1
        self.smooth_f = 0

        self.bc_domain = bc_domain
       
        # Dirichlet BC for the forward and the adjoint operators
        self._state_bcs = dl.DirichletBC(self.Vh_state, bc_exp, self.bc_domain)
        p_exp  = dl.Constant(0.0)
        self._adjoint_bcs = dl.DirichletBC(self.Vh_state, p_exp, self.bc_domain)

        v = dl.TestFunction(self.Vh_state)
        u = dl.TrialFunction(self.Vh_state)
        
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
                #print('t: ', t)

                obs_rhs = rhs.solution_at_time(t)
                obs_rhs = dl.Function(self.Vh_state, obs_rhs)


                #obs_rhs =  dl.Function(self.Vh_state, rhs_i)
                #obs_rhs = dl.project(obs_rhs*self.obs_point_src, self.Vh_state)
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
        out = dl.Function(self.Vh_parameter).vector()
        product = dl.Function(self.Vh_parameter).vector()
        #self.M.init_vector(product, 0)
        out.zero()

        Ph = self.Vh_parameter
        Vh = self.Vh_state

        k_test = dl.TestFunction(Ph)
        v_trial = dl.TrialFunction(Vh)
        t = self.t_init
        k = x
        k_func = dl.Function(self.Vh_parameter, k)

        while t < self.t_final -0.5 *self.dt:
            t += self.dt

            u_func = self.fwd_sol.solution_at_time(t)

            u_func = dl.Function(self.Vh_state, u_func)

            Ct_i = dl.assemble(self.dt*dl.exp(k_func)*k_test*\
                              dl.inner(dl.nabla_grad(u_func),\
                              dl.nabla_grad(v_trial))*dl.dx)

            dummy = dl.Vector()
            Ct_i.init_vector(dummy,0)
            self._adjoint_bcs.zero_columns(Ct_i, dummy)
        
            product.zero()
            dp_current = self.adj_sol.solution_at_time(t)

            #[dp[0] for dp in self.adj_sol if np.isclose(dp[1], t)]
            #assert len(dp_current) == 1
            #dp_current = dp_current[0]

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
    
    def max(self):
        # find the max value 
        max=-np.inf
        for sol in self._sol:
            max_temp = np.max(sol.max())
            if max_temp > max:
                max = max_temp
        return max


class CUQIpyFwd:
    def __init__(self, *args, **kwargs):
        self.fwd = TimeDependantHeat(*args, **kwargs)

    def forward(self, k):
        self.fwd.solveFwd(k.vector())
        #extract obs in an array
        obs = np.zeros((len(self.fwd.obs_locations), len(self.fwd.obs_times)) )
        for i, obs_t in enumerate(self.fwd.obs_times):
            #print ('obs_t: ', obs_t)
            #print ('i', i )
            sol_t = self.fwd.fwd_sol.solution_at_time(obs_t)
            sol_t_func = dl.Function(self.fwd.Vh_state, sol_t)
            for j, loc in enumerate(self.fwd.obs_locations):
                obs[j, i] = sol_t_func(dl.Point(loc,0))
        return obs
    
    def compute_rhs(self, dirc):
        ts = self.fwd.obs_times
        locs = self.fwd.obs_locations
        dirc = dirc.reshape((len(locs), len(ts)))
        rhs = TimeDependantSolution()
        for i, t in enumerate(ts):
            vals = dirc[:,i]
            rhs_fun_i = dl.interpolate(Dirac(loc=fwd.obs_locations,                   
                f=fwd.f_dirac, vals=vals, L=fwd.L, degree=1,
                smooth_f=fwd.smooth_f), fwd.Vh_state)
            rhs.add(rhs_fun_i.vector(), t)
        return rhs
        
    def gradient(self, dirc, k):
        Vh_parameter = self.fwd.Vh_parameter
        self.fwd.solveFwd(k.vector())
        rhs = self.compute_rhs(dirc)
        self.fwd.solveAdj(k.vector(), rhs)
        grad = self.fwd.evalGradientParameter(k.vector())
        return dl.Function(Vh_parameter, grad)

#https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-for-a-dataset
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# approximate dirac at loc = [2, 3]
def dirac_approx(x, loc, vals, f=1, smooth_f=20):
    h = f
    funval = sum([np.exp(-(x-loc_i)**2/h**2)*vals[i] for i, loc_i in enumerate(loc)])
    #conv = np.convolve(funval)
    return smooth(funval,  int(f*smooth_f))

# Fenics expression for the dirac
class Dirac_old(dl.UserExpression):
    def __init__(self, loc, f, vals, L, smooth_f, **kwargs):
        super().__init__(**kwargs)
        grid = np.linspace(0, L, 1000)
        funvals = dirac_approx(grid, loc=loc, vals=vals, f=f, smooth_f=smooth_f)
        self.funvals = funvals
        self.func = interp1d(
            grid, funvals, kind='linear',
            fill_value='extrapolate')


    def eval(self, values, x):
        values[0] = self.func(x[0])

    def value_shape(self):
        return ()

class Dirac(dl.UserExpression):
    def __init__(self, loc, f, vals, L, smooth_f, **kwargs):
        super().__init__(**kwargs)
        grid = np.linspace(0, L, 1000)
        self.loc = np.array(loc)
        self.vals = np.array(vals)
        self.f = f



    def eval(self, values, x):
        # if x is close to loc, return the corresponding value
        idx = np.abs(x[0]-self.loc) < self.f
        if np.any(idx):
            values[0] = self.vals[idx][0]
        else:
            values[0] = 0


    def value_shape(self):
        return ()
    
if __name__ == "__main__":
    # CREATE 1D mesh
    L = 50
    nx = 100

    # Start and final time
    t_init = 0
    t_final = 12 # 0.01
    dt = 1
    
    bc_tol = 1E-14
    class LeftBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):  
            return on_boundary and abs(x[0]) < bc_tol
        
    bc_domain = LeftBoundary()  
    bc_exp = dl.Expression('1', degree=1)  


    # data

    # CREATE TimeDependantHeat object
    locations = [ 6, 15]#[0.1, 0.2, 0.3, 0.4, 0.5]
    u0 = dl.Expression('0', degree=1)
    f = dl.Expression('0', degree=1)
    fwd = TimeDependantHeat(L, nx, t_init, t_final, dt, u0, f, bc_exp, bc_domain, locations,)
    fwd.obs_times = fwd.sim_times[[5, 10]]

    # Create parameter vector as a sine function
    #k = dl.Expression('sin(2*pi*x[0])+1.1', degree=1)
    k = dl.Expression('std::log(1.1 +sin((x[0]/L*2)*2*pi))', L=L, degree=1)
    k_fun = dl.interpolate(k, fwd.Vh_parameter)
    k_vec = k_fun.vector()

    # Solve the fwd problem
    # time the fwd solve
    start = time.time()
    fwd.solveFwd(k_vec)
    end = time.time()
    print('fwd solve time: ', end-start)

    fwd.plotTimes(fwd.sim_times, fwd.fwd_sol)
    plt.ylim(-0.1, fwd.fwd_sol.max()*1.1)
    plt.title('Forward solution, for exact parameter')

    data_exact = fwd.fwd_sol.copy()
    fwd.fwd_sol.clear()

    k_const = dl.Expression('0', degree=1)
    k_const = dl.interpolate(k_const, fwd.Vh_parameter).vector()

    # Solve the fwd problem
    fwd.solveFwd(k_const)
    plt.figure()
    fwd.plotTimes(fwd.sim_times, fwd.fwd_sol)
    plt.ylim(-0.01, fwd.fwd_sol.max()*1.1)
    plt.title('Forward solution, for constant parameter')

    # Solve the adj problem
    def compute_rhs_old(fwd):
        rhs = TimeDependantSolution()
        for i in range(len(fwd.fwd_sol)):
            rhs_i = fwd.fwd_sol[i][0].copy()
            rhs_i.axpy(-1, data_exact[i][0])
            rhs_i *= -dt
            rhs.add(rhs_i, fwd.fwd_sol[i][1])
        return rhs
    
    def compute_rhs(fwd):
        # k from numpy array to dolfin vector


        rhs = TimeDependantSolution()
        for i in range(len(fwd.fwd_sol)):
            if np.any(np.isclose(fwd.sim_times[i], fwd.obs_times)):
                u_i = fwd.fwd_sol[i][0]
                data_i = data_exact[i][0]
                diff_i = u_i.copy()
                diff_i.axpy(-1, data_i)
                vals = []
                diff_i_func = dl.Function(fwd.Vh_state, diff_i)
                for loc in fwd.obs_locations:
                    vals.append(diff_i_func(dl.Point(loc,0)))
                diff_i_func_pts = dl.interpolate(Dirac(loc=fwd.obs_locations,                   
                 f=fwd.f_dirac, vals=vals, L=fwd.L, degree=1, smooth_f=fwd.smooth_f), fwd.Vh_state)

                #diff_i_func = dl.Function(Vh_state, diff_i)
                #diff_i_func_pts = dl.project(diff_i_func*fwd.obs_point_src, Vh_state)
                diff_i = diff_i_func_pts.vector()

                #print('t: ', fwd.sim_times[i])
                rhs.add(-1*dt*diff_i, fwd.sim_times[i])
                # No multiplication by M

                #cost += dt*0.5*diff_i.inner(fwd.M*diff_i)
                print(fwd.sim_times[i])
        return rhs
    # plot rhs
    plt.figure()
    rhs = compute_rhs(fwd)
    fwd.plotTimes(fwd.obs_times, rhs)
    plt.title('rhs')
    plt.ylim([-0.01, 0.1])
    
    # time the adj solve
    start = time.time()
    fwd.solveAdj(k_const, rhs)
    end = time.time()
    print('adj solve time: ', end-start)

    # Plot the adjoint solution
    plt.figure()
    fwd.plotTimes(fwd.sim_times, fwd.adj_sol)
    plt.ylim([-0.01, fwd.adj_sol.max()*1.1])
    plt.title('Adjoint solution, for constant parameter')


    # Compute the gradient
    grad = fwd.evalGradientParameter(k_const)


    # cost function
    def cost(k):
        # k from numpy array to dolfin vector
        k_v = dl.Function(fwd.Vh_parameter).vector()
        k_v[:] = k[:]
        fwd.solveFwd(k_v)
        cost = 0
        for i in range(len(fwd.fwd_sol)):
            if np.any(np.isclose(fwd.sim_times[i], fwd.obs_times)):
                u_i = fwd.fwd_sol[i][0]
                data_i = data_exact[i][0]
                diff_i = u_i.copy()
                diff_i.axpy(-1, data_i)
                vals = []
                diff_i_func = dl.Function(fwd.Vh_state, diff_i)
                for loc in fwd.obs_locations:
                    vals.append(diff_i_func(dl.Point(loc,0)))
                diff_i_func_pts = dl.interpolate(Dirac(loc=fwd.obs_locations,                   
                 f=fwd.f_dirac, vals=vals, L=fwd.L, smooth_f=fwd.smooth_f, degree=1), fwd.Vh_state)

                #diff_i_func = dl.Function(Vh_state, diff_i)
                #diff_i_func_pts = dl.project(diff_i_func*fwd.obs_point_src, Vh_state)
                diff_i = diff_i_func_pts.vector()

                #print('t: ', fwd.sim_times[i])
                cost += dt*0.5*diff_i.inner(fwd.M*diff_i)
        return cost
    verify_grad_const = True
    #%%
    if verify_grad_const:
        # gradient check using scipy.optimize.approx_fprime
        from scipy.optimize import approx_fprime
        grad_scipy = approx_fprime(k_const.get_local(), cost, 1e-12)
        grad_scipy_const_func = dl.Function(fwd.Vh_parameter)
        grad_scipy_const_func.vector()[:] = grad_scipy[:]

        # Plot the gradients
        plt.figure()
        dl.plot(dl.Function(fwd.Vh_parameter, grad), label='adj')
        dl.plot(grad_scipy_const_func, label='scipy')
        plt.title('gradient at constant parameter')
        plt.legend()
    #%%

    verify_grad_true = False
    if verify_grad_true: 
        # compute gradient at true
        fwd.solveFwd(k_vec)
        rhs = compute_rhs(fwd)
        fwd.solveAdj(k_vec, rhs)
        grad_true = fwd.evalGradientParameter(k_vec)
        plt.figure()
        dl.plot(dl.Function(fwd.Vh_parameter, grad_true))
        plt.title('gradient at true')
        #
        grad_scipy_true = approx_fprime(k_vec.get_local(), cost, 1e-6)
        plt.figure()
        plt.plot(grad_scipy_true[::-1]) 
        plt.title('scipy gradient at true')
    #%%
    verify_grad_random = True
    if verify_grad_random: 
        random_k = dl.Vector()
        fwd.M.init_vector(random_k, 0)
        random_k.set_local(np.random.rand(random_k.local_size()))
        # compute gradient at random
        # time adj gradient
        start = time.time()
        fwd.solveFwd(random_k)
        rhs = compute_rhs(fwd)
        fwd.solveAdj(random_k, rhs)
        grad_random = fwd.evalGradientParameter(random_k)
        end = time.time()
        print('adj gradient time: ', end-start)



        # time scipy gradient
        start = time.time()
        grad_scipy_random = approx_fprime(random_k.get_local(), cost, 1e-9)
        grad_scipy_random_func = dl.Function(fwd.Vh_parameter)
        grad_scipy_random_func.vector()[:] = grad_scipy_random[:]
        end = time.time()
        print('scipy gradient time: ', end-start)

        plt.figure()
        dl.plot(dl.Function(fwd.Vh_parameter, grad_random), label='adj')
        dl.plot(grad_scipy_random_func, label='scipy')
        plt.legend()
        plt.title('gradient at random')

    #%%


#    #%% Test CUQIpyFwd
#    cuqipy_fwd = CUQIpyFwd(fwd.mesh, fwd.Vh_parameter, fwd.Vh_state, t_init, t_final, dt, u0, f, locations, fwd.obs_times)
#    #obs = cuqipy_fwd.forward(k)
#    #grad = cuqipy_fwd.gradient(k_vec)
#    
#    from advection_diffusion_inference_utils import plot_time_series
#    #plot_time_series(cuqipy_fwd.fwd.obs_times, cuqipy_fwd.fwd.obs_locations, obs)
#    
#    #%% domain geometry
#    from cuqipy_fenics.geometry import FEniCSContinuous,\
#    MaternKLExpansion
#    G_FEM = FEniCSContinuous(Vh_parameter)
#        
#    # The KL parameterization
#    G_KL = MaternKLExpansion(G_FEM, length_scale=0.2, num_terms=2)
#    
#    #%% Create range geometry
#    from cuqi.geometry import Continuous2D
#    G_cont = Continuous2D((locations, fwd.obs_times))
#    
#    # %%
#    # forward model
#    A = cuqi.model.Model(forward=cuqipy_fwd.forward, gradient=cuqipy_fwd.gradient, range_geometry=G_cont, domain_geometry=G_KL)
#    
#    # %% 
#    prior = cuqi.distribution.Gaussian(mean=0.0, cov=50**2, geometry=G_KL)
#    #%%
#    np.random.seed(0)
#    x_true = prior.sample()
#    x_true.plot(title='True parameter')
#    data = A(x_true)
#    plt.figure()
#    plot_time_series(cuqipy_fwd.fwd.obs_times, cuqipy_fwd.fwd.obs_locations, data.funvals)
#    
#    #%% 
#    # Likelihood
#    s_noise = 0.003
#    y = cuqi.distribution.Gaussian(A(prior), s_noise**2, geometry=G_cont)
#    
#    # %%
#    noisy_data = y(prior=x_true).sample()
#    plt.figure()
#    plot_time_series(cuqipy_fwd.fwd.obs_times, cuqipy_fwd.fwd.obs_locations, noisy_data.funvals)
#    norm_rel_noise = np.linalg.norm(noisy_data.funvals-data.funvals)/np.linalg.norm(data.funvals)
#    plt.title('Relative noise norm: {:.2f}%'.format(norm_rel_noise*100))
#    # %%
#    # posterior
#    joint = cuqi.distribution.JointDistribution(prior, y)
#    posterior = joint(y=noisy_data)
#    # %%
#    # sample from the posterior using NUTS
#    from cuqi.sampler import NUTS
#    sampler = NUTS(posterior, max_depth=4)
#    #samples = sampler.sample(10, 5 )
#    # %%
#    #check posterior grad
#    np.random.seed(2)
#    prior_sample_i = prior.sample()
#    g_adj = posterior.gradient(prior_sample_i)
#    
#    #%%
#    from scipy.optimize import approx_fprime
#    def cost(x):
#        return posterior.logpdf(x)
#    g_FD = approx_fprime(prior_sample_i, cost, 1e-8)
#
#    plt.figure()
#    g_adj.plot(title='adjoint gradient')
#    plt.figure()
#    dl.plot(G_KL.par2fun(g_FD))
#    plt.title('FD gradient')
## %%
##
#
#
#fenics_dirac = Dirac(loc=[10], vals=[1], f=10, degree=1, L=fwd.L, smooth_f=1)
#fenics_dirac_func = dl.interpolate(fenics_dirac, fwd.Vh_state)
#dl.plot(fenics_dirac_func)
#
#
#
##%%
#
##plot dirac
#
#x = np.linspace(0, 100, 1000)
#plt.figure()
#plt.plot(x, dirac_approx(x, loc=[20,40], vals=[1,2], f=1))
#plt.show()


# %%
