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
from my_utils import plot_time_series, save_experiment_data

# WHICH ANIMAL AND EAR TO USE
args_animal = 'm1'
args_ear = 'l'

# ADDITIONAL PARAMETERS
factor = .5 # factor for refining the grid and the time steps
Pec = 10 # Peclet number
c_2_max = 100 # Maximum diffusion coefficient
c_2_min = 100 # Minimum diffusion coefficient
args_unknown_par_type = 'constant' # Type of diffusivity profile: 
                                   #'constant', 'step', 'smooth'
manufactured = False # If True, use a manufactured solution
                     # to verify the implementation

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
manufactured_tag = 'manuf' if manufactured else 'not_manuf'
c_2_tag = 'c2_'+str(c_2_max)+'_'+str(c_2_min) if args_unknown_par_type != 'constant' else 'c2_'+str(c_2_max)
expr_tag = 'expr'+args_animal+'_'+args_ear+'_'+str(Pec)+'_'+\
    c_2_tag+'_'+manufactured_tag+'_'+args_unknown_par_type+'_factor'+str(factor)

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
n_grid = int(400*factor) # Number of solution nodes
h = L/(n_grid+1) # Space step size
grid = np.linspace(h, L-h, n_grid)

# GRID FOR THE DIFFUSION COEFFICIENT
n_grid_c = 20
h_c = L/(n_grid_c+1) 
grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)

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
DA_a = (np.diag(np.ones(n_grid-1), 1) +\
        -np.diag(np.ones(n_grid), 0)) * (a/h)

# SET UP THE RANGE GEOMETRY
G_cont2D = Continuous2D((locations, times))

# MODEL0- CONSTANT DIFFUSION COEFFICIENT: VERIFY THE IMPLEMENTATION
if manufactured:
    tau_max = 0.5 # Final time in sec
    cfl = 0.01/factor # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)

    # Solution
    def exact_sol(x, t):
        return np.exp(-Pec*t)*np.sin(np.pi*x/L)

    # Source term
    def g_manufactured(x, t, c):
        return -Pec*np.exp(-Pec*t)*np.sin(np.pi*x/L) +\
        c*(np.pi/L)**2*np.exp(-Pec*t)*np.sin(np.pi*x/L)+\
        a*np.pi/L*np.exp(-Pec*t)*np.cos(np.pi*x/L)

    # Initial condition
    def initial_condition_manufactured(x):
        return np.sin(np.pi*x/L)
    
    def g_numerical(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = -exact_sol(0, tau_current)*(-2*c/h**2 - -1*a/h)
        f_array[1] = -exact_sol(0, tau_current)*(c/h**2)
        f_array[-1] = -exact_sol(L, tau_current)*(c/h**2 - a/h)
        f_array[-2] = -exact_sol(L, tau_current)*(-2*c/h**2 - (-1)*a/h)
        f_array += g_manufactured(grid, tau_current, c)
        return f_array
    
    initial_condition = initial_condition_manufactured(grid)

    # Differential operator (constant diffusion coefficient case)
    DD_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
    np.diag(np.ones(n_grid-1), -1) +
    np.diag(np.ones(n_grid-1), 1) ) / h**2

    # Operator Advection-diffusion
    D_c_const = lambda c: DD_c_const(c) - DA_a
    
    # PDE form (constant diffusion coefficient case)
    def PDE_form_const(c, tau_current):
        return (D_c_const(c), g_numerical(c, tau_current), initial_condition)

    # CUQIpy PDE object (constant diffusion coefficient case)
    PDE = TimeDependentLinearPDE(PDE_form_const, tau, grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                time_obs=times) 

    # Domain geometry (constant diffusion coefficient case)
    G_D = MappedGeometry( Discrete(1),  map=lambda x: x**2 )

    # PDE forward model (constant diffusion coefficient case)
    A = PDEModel(PDE, range_geometry=G_cont2D,domain_geometry=G_D)

    exact_sol_array = np.empty([len(grid), len(tau)])
    for i, tau_i in enumerate(tau): 
        exact_sol_array[:, i] = exact_sol(grid, tau_i)

# MODEL1- CONSTANT DIFFUSION COEFFICIENT: CREATE THE FORWARD MODEL
else:

    tau_max = 30*60 # Final time in sec
    cfl = 4/factor # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)

    # INITIAL CONDITION
    initial_condition = np.zeros(n_grid)

    if args_unknown_par_type == 'constant':
        def g_const(c, tau_current):
            f_array = np.zeros(n_grid)
            # exact_sol(0, tau_current)*(-2*c/h**2 + a/h)
            f_array[0] = c/h**2*np.interp(tau_current, times, data_bc)
            
            #f_array[0] = -np.interp(tau_current, times, data_bc)*(-2*c/h**2 - -1*a/h)
            #f_array[1] = -np.interp(tau_current, times, data_bc)*(c/h**2)
            #f_array[-2] = -np.interp(tau_current, times, data_bc_end)*(c/h**2 - a/h)
            #f_array[-1] = -np.interp(tau_current, times, data_bc_end)*(-2*c/h**2 - (-1)*a/h)
            return f_array

        # Differential operator (constant diffusion coefficient case)
        DD_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
        np.diag(np.ones(n_grid-1), -1) +
        np.diag(np.ones(n_grid-1), 1) ) / h**2

        # Operator Advection-diffusion
        D_c_const = lambda c: DD_c_const(c) - DA_a
        
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
            #f_array[0] = c[0]/h**2*np.interp(tau_current, times, data_bc)
            #TODO: in f_array[1] and f_array[-2] the interpolation for c is not correct (afterthought, I think it is correct, c need to be the ends c here: corresponds to the c multiplying first and last column of the differential operator) 
            f_array[0] = -np.interp(tau_current, times, data_bc)*(-2*c[0]/h**2 - -1*a/h)
            f_array[1] = -np.interp(tau_current, times, data_bc)*(c[0]/h**2)
            #f_array[-2] = -np.interp(tau_current, times, data_bc_end)*(c[-1]/h**2 - a/h)
            #f_array[-1] = -np.interp(tau_current, times, data_bc_end)*(-2*c[-1]/h**2 - (-1)*a/h)
            return f_array
        
        # Differential operator (varying in space diffusion coefficient case)
        Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
        vec = np.zeros(n_grid)
        vec[0] = 1
        Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
        Dx /= h # FD derivative matrix
        
        D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx - DA_a

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

exact_x = CUQIarray(exact_x, is_par=False, geometry=G_D)

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

# SAVE THE EXACT DATA, LOCATION, TIME, AND EXPERIMENT PARAMETERS
np.savez(dir_name+'exact_data.npz', exact_data=exact_data, locations=locations,
         times=times, exact_x=exact_x, experiment_par=expr_tag)

# IF MANUFACTURED SOLUTION, PLOT THE SOLUTION
if manufactured:
    plt.figure()
    plt.plot(grid, exact_sol_array)
    plt.title('Exact analytical sol, a='+str(a)+', Pec='+str(Pec))
    plt.savefig(dir_name+'exact_analytical_sol.png')

# IF MANUFACTURED SOLUTION, PLOT THE DIFFERENCE BETWEEN THE SOLUTIONS
if manufactured:
    plt.figure()
    plt.plot(grid, (exact_sol_array-u))
    plt.title('Difference between exact and numerical sol, a='+str(a)+\
              ', Pec='+str(Pec)+'\n'+"Relative error: "+\
                str(np.linalg.norm(exact_sol_array-u)/np.linalg.norm(exact_sol_array)))
    plt.savefig(dir_name+'diff_exact_numerical.png')

