#%% MODELING FLOW IN THE EAR AQUEDUCTS: ADVECTION-DIFFUSION FORWARD PROBLEM

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from cuqi.geometry import Continuous1D, Discrete,\
    MappedGeometry, Continuous2D
from cuqi.pde import TimeDependentLinearPDE
from cuqi.model import PDEModel
from cuqi.array import CUQIarray
from my_utils import plot_time_series

# WHICH ANIMAL AND EAR TO USE
args_animal = 'm2'
args_ear = 'r'

# ADDITIONAL PARAMETERS
Pec = 10 # Peclet number
c_2_max = 1000 # Maximum diffusion coefficient
c_2_min = 400 # Minimum diffusion coefficient
args_unknown_par_type = 'constant' # Type of diffusivity profile: 
                                   #'constant', 'step', 'smooth'

# READ ANIMAL DATA, (LOCATION)
dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+args_animal+'_'+args_ear+'_distances.csv')
locations = dist_file['distance microns'].values[:5]
locations_real = locations

# READ ANIMAL DATA, (CONCENTRATION AND TIME)
constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+args_animal+'_'+args_ear+'_parsed.csv')
times = constr_file['time'].values*60
data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()

# EXTRACT THE DATA FOR THE FIRST LOCATION TO USE AS BOUNDARY CONDITION
data_bc = data.reshape([len(locations_real), len(times)])[0,:]

# CREATE EXPERIMENT TAG
expr_tag = 'expr'+args_animal+'_'+args_ear+'_'+str(Pec)

# CREATE OUTPUT DIRECTORY
dir_name = 'output_advection_diffusion/'+expr_tag+'/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
else:
    raise Exception('Output directory already exists')

# SAVE THIS SCRIPT IN THE OUTPUT DIRECTORY
os.system('cp '+__file__+' '+dir_name+'/')

# SET THE PDE PARAMETERS
L = 500
n_grid = 100 # Number of solution nodes
h = L/(n_grid+1) # Space step size
grid = np.linspace(h, L-h, n_grid)

# GRID FOR THE DIFFUSION COEFFICIENT
n_grid_c = 20
h_c = L/(n_grid_c+1) 
grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)

tau_max = 30*60 # Final time in sec
cfl = 5 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
dt_approx = cfl*h**2 # Defining approximate time step size
n_tau = int(tau_max/dt_approx)+1 # Number of time steps
tau = np.linspace(0, tau_max, n_tau)

# CREATE THE EXACT DIFFUSION COEFFICIENT
if args_unknown_par_type == 'constant':
    exact_x = c_2_max

# if the unknown parameter is varying in space (step function)
elif args_unknown_par_type == 'step':
    exact_x = np.zeros(n_grid_c)
    exact_x[0:n_grid_c//2] = c_2_max
    exact_x[n_grid_c//2:] = c_2_min

# if the unknown parameter is varying in space (smooth function)
elif args_unknown_par_type == 'smooth':
    low = c_2_max
    high = c_2_min
    exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low  

# SET UP THE ADVECTION COEFFICIENT
min_diffusion = exact_x if args_unknown_par_type == 'constant' else min(exact_x)
a = Pec * min_diffusion / L

## DIFFERENTIAL OPERATOR FOR ADVECTION
DA_a = (a*-np.diag(np.ones(n_grid), 0) -\
        np.diag(np.ones(n_grid-1), -1)) / (h)

# INITIAL CONDITION
initial_condition = np.zeros(n_grid)

# SET UP THE RANGE GEOMETRY
G_cont2D = Continuous2D((locations, times))

# MODEL1- CONSTANT DIFFUSION COEFFICIENT: CREATE THE FORWARD MODEL
if args_unknown_par_type == 'constant':
    def g_const(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = c/h**2*np.interp(tau_current, times, data_bc)
        return f_array
    
    # Differential operator (constant diffusion coefficient case)
    DD_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
    np.diag(np.ones(n_grid-1), -1) +
    np.diag(np.ones(n_grid-1), 1) ) / h**2

    # Operator Advection-diffusion
    D_c_const = lambda c: DD_c_const(c) + DA_a
    
    # PDE form (constant diffusion coefficient case)
    def PDE_form_const(c, tau_current):
        return (D_c_const(c), g_const(c, tau_current), initial_condition)

    # CUQIpy PDE object (constant diffusion coefficient case)
    PDE = TimeDependentLinearPDE(PDE_form_const, tau, grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                time_obs=times) 

    # Domain geometry (constant diffusion coefficient case)
    G_D =  MappedGeometry( Discrete(1),  map=lambda x: x**2 )

    # PDE forward model (constant diffusion coefficient case)
    A = PDEModel(PDE, range_geometry=G_cont2D,domain_geometry=G_D)

# MODEL2- VARYING IN SPACE DIFFUSION COEFFICIENT: CREATE THE FORWARD MODEL
else:

    # Source term (varying in space diffusion coefficient case)
    def g_var(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = c[0]/h**2*np.interp(tau_current, times, data_bc)
        return f_array
    
    # Differential operator (varying in space diffusion coefficient case)
    Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
    vec = np.zeros(n_grid)
    vec[0] = 1
    Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
    Dx /= h # FD derivative matrix
    
    D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx + DA_a
    
    def PDE_form_var(c, tau_current):
        c = np.interp(grid_c_fine, grid_c, c)
        return (D_c_var(c), g_var(c, tau_current), initial_condition)
    
    PDE = TimeDependentLinearPDE(PDE_form_var, tau, grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                time_obs=times) 
    
    # Domain geometry
    G_D =  MappedGeometry( Continuous1D(grid_c), map=lambda x: x**2 )#TEMP: one map
    
    A = PDEModel(PDE, range_geometry=G_cont2D, domain_geometry=G_D)

exact_x = CUQIarray(exact_x, geometry=G_D)

# SOLVE THE FORWARD PROBLEM
PDE.assemble(exact_x)
u, _ = PDE.solve()
exact_data = A(exact_x)

# PLOT AND SAVE THE SOLUTION (OVER SPACE)
plt.plot(grid, u)
plt.title('Solution at different times, a='+str(a)+', Pec='+str(Pec))
plt.savefig(dir_name+'sol_space.png')

# PLOT AND SAVE THE SOLUTION (OVER TIME)
plt.figure()
plot_time_series(times, locations, exact_data.reshape([len(locations), len(times)]))
plt.title('Exact sol, a='+str(a)+', Pec='+str(Pec))
plt.savefig(dir_name+'exact_data.png')
