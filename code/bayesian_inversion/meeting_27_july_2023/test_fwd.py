## Imports 
#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cuqi.pde import TimeDependentLinearPDE
import sys
sys.path.append('../')
from my_utils import plot_time_series
np.random.seed(1)

# Arguments
class args:
    animal = 'm1'
    ear = 'l'

print('Using default arguments: animal = '+str(args.animal)+', ear = '+str(args.ear))

## Read distance file
dist_file = pd.read_csv('../../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_distances.csv')
locations = dist_file['distance microns'].values[:5]
print(locations)

## Read concentration file
constr_file = pd.read_csv('../../../data/parsed/CT/20210120_'+args.animal+'_'+args.ear+'_parsed.csv')
data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
print(data)
times = constr_file['time'].values*60
print(times)

def solve_heat(n, c_square):
    ## Set PDE parameters
    L = 500
    n_grid = n   # Number of solution nodes
    h = L/(n_grid+1)   # Space step size
    grid = np.linspace(h, L-h, n_grid)
    
    tau_max = 30*60 # Final time in sec
    cfl = 5 # The cfl condition to have a stable solution
             # the method is implicit, we can choose relatively large time steps 
    dt_approx = cfl*h**2 # Defining approximate time step size
    n_tau = int(tau_max/dt_approx)+1 # Number of time steps
    tau = np.linspace(0, tau_max, n_tau)
    
    ## Source term (constant diffusion coefficient case)
    def g_const(c, tau_current):
        f_array = np.zeros(n_grid)
        f_array[0] = c/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
        return f_array
    
    ## Differential operator (constant diffusion coefficient case)
    D_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
    np.diag(np.ones(n_grid-1), -1) +
    np.diag(np.ones(n_grid-1), 1) ) / h**2
    
    ## Initial condition
    initial_condition = np.zeros(n_grid) 
    
    ## PDE form (constant diffusion coefficient case)
    def PDE_form_const(c, tau_current):
        return (D_c_const(c), g_const(c, tau_current), initial_condition)
    
    ## CUQIpy PDE object (constant diffusion coefficient case)
    PDE_const = TimeDependentLinearPDE(PDE_form_const, tau, grid_sol=grid,
                                 method='backward_euler', 
                                 grid_obs=locations,
                                time_obs=times) 
    
    PDE_const.assemble(c_square)
    u, _ = PDE_const.solve()
    u_obs = PDE_const.observe(u)

    return u_obs

c_square = 400
n = 150

loop_over_n = False
if loop_over_n:
    u_obs_1000 = solve_heat(1000, c_square)

for c_square in [ 4, 40, 400, 4000, 40000]:
    print(n)
    u_obs_n = solve_heat(n, c_square)

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    plt.sca(ax[0]) 
    plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )
    plt.title('Data')

    plt.sca(ax[1])
    plot_time_series( times, locations, u_obs_n)# data.reshape([len(locations), len(times)]) )
    title_str = 'Solution for n = '+str(n)+', diffusivity = '+str(c_square)
    if loop_over_n:
        title_str += ', error compared\nto the 1000 node sol = '+str(np.linalg.norm(u_obs_n-u_obs_1000)/np.linalg.norm(u_obs_1000))
    plt.title(title_str)   

    plt.savefig(args.animal+args.ear+'_heat_n_'+str(n)+'_c_square_'+str(c_square)+'.png', dpi=300)
