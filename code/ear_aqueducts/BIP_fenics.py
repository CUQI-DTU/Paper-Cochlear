#%% IMPORTS
from fenics_test import TimeDependantHeat, CUQIpyFwd
from advection_diffusion_inference_utils\
    import read_data_files,\
    plot_time_series
from cuqipy_fenics.geometry import FEniCSContinuous,\
MaternKLExpansion
from cuqi.geometry import Continuous2D
from cuqi.model import Model
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.sampler import NUTS
import matplotlib.pyplot as plt
    
import dolfin as dl
import numpy as np

#%% SET UP ARGS
class Args:
    def __init__(self):
        self.num_ST = 0
        self.animal = 'm1'
        self.ear = 'l'
        self.num_CA = 5
        self.case_synthetic = True

args = Args()

#%% READ DATA
real_times, real_locations, real_data, real_std_data = read_data_files(args)

#%% CREATE CUQIpyFwd OBJECT
# PDE mesh
L = real_locations[-1]*1.01
n_grid =int(L/5)   # Number of solution nodes
mesh = dl.IntervalMesh(n_grid, 0, L)

# PDE time step
tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
dt_approx = cfl*np.floor(L/n_grid)**2 # Defining approximate time step size
n_tau = int(tau_max/dt_approx)+1 # Number of time steps
tau = np.linspace(0, tau_max, n_tau)
tau = np.linspace(0, tau_max, int(1800/100)+1)

# PDE initial and boundary conditions
u0 = dl.Expression('0', degree=1)
f = dl.Expression('0', degree=1)
bc_tol = 1E-14
class LeftBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):  
        return on_boundary and abs(x[0]) < bc_tol
    
bc_domain = LeftBoundary()  
bc_exp = dl.Expression('4', degree=1)  
CUQIpy_fwd = CUQIpyFwd(L, n_grid, 0, tau_max,
                        tau[-1]-tau[-2], u0, f, bc_exp, bc_domain, real_locations)
CUQIpy_fwd.fwd.obs_times = real_times

#%% Create forward model

# Domain geometry
Vh_parameter = CUQIpy_fwd.fwd.Vh_parameter
G_FEM = FEniCSContinuous(Vh_parameter)
    
# The KL parameterization
G_KL = MaternKLExpansion(G_FEM, length_scale=50, num_terms=2)

# Create range geometry
G_cont = Continuous2D((real_locations, real_times))

# Forward model
#A = Model(forward=CUQIpy_fwd.forward, gradient=CUQIpy_fwd.gradient, range_geometry=G_cont, domain_geometry=G_KL)

A = Model(forward=CUQIpy_fwd.forward, range_geometry=G_cont, domain_geometry=G_KL)


# %% Test applying the forward model
Diff_100 = dl.Function(A.domain_geometry.function_space)
Diff_100.vector()[:] = np.log(100)
y_test = A(Diff_100, is_par=False)
#%%
plot_time_series(
    CUQIpy_fwd.fwd.obs_times,
    CUQIpy_fwd.fwd.obs_locations,
    y_test.reshape(
        (len(CUQIpy_fwd.fwd.obs_locations),
         len(CUQIpy_fwd.fwd.obs_times))
         )
)
# %% Create prior distribution
prior = Gaussian(mean=0.0, cov=50**2, geometry=G_KL)
#
np.random.seed(0)
x_true = prior.sample()
x_true.plot(title='True parameter')
data = A(x_true)
plt.figure()
plot_time_series(CUQIpy_fwd.fwd.obs_times, CUQIpy_fwd.fwd.obs_locations, data.funvals)

#%% Likelihood
s_noise = 0.003
y = Gaussian(A(prior), s_noise**2, geometry=G_cont)

# %% Create noisy data
if args.case_synthetic:
    noisy_data = y(prior=x_true).sample()
    plt.figure()
    plot_time_series(CUQIpy_fwd.fwd.obs_times, CUQIpy_fwd.fwd.obs_locations, noisy_data.funvals)
    norm_rel_noise = np.linalg.norm(noisy_data.funvals-data.funvals)/np.linalg.norm(data.funvals)
    plt.title('Relative noise norm: {:.2f}%'.format(norm_rel_noise*100))
    print("This is synthetics data case")
else:
    noisy_data = real_data
# %%
# posterior
joint = JointDistribution(prior, y)
posterior = joint(y=noisy_data)
# %%
# sample from the posterior using NUTS

sampler = NUTS(posterior, max_depth=4)
#samples = sampler.sample(10, 5 )
# %%
#check posterior grad
np.random.seed(2)
prior_sample_i = prior.sample()
g_adj = posterior.gradient(prior_sample_i)

#%%
from scipy.optimize import approx_fprime
def cost(x):
    return posterior.logpdf(x)
g_FD = approx_fprime(prior_sample_i, cost, 1e-8)

plt.figure()
g_adj.plot(title='adjoint gradient')
plt.figure()
dl.plot(G_KL.par2fun(g_FD))
plt.title('FD gradient')
