#%% 1- IMPORTS
import numpy as np
from cuqi.pde import TimeDependentLinearPDE
from cuqi.geometry import Continuous1D, MappedGeometry, Continuous2D
from cuqi.model import PDEModel
from cuqi.array import CUQIarray
import matplotlib.pyplot as plt

#%% 2- Set up time dependent linear PDE forward model
L = 5
n_grid =100   # Number of solution nodes
h = L/(n_grid+1)   # Space step size
grid = np.linspace(h, L-h, n_grid)
tau_max = 1 # Final time in sec
cfl = 4 # The cfl condition to have a stable solution
         # the method is implicit, we can choose relatively large time steps 
dt_approx = cfl*h**2 # Defining approximate time step size
n_tau = int(tau_max/dt_approx)+1 # Number of time steps
tau = np.linspace(0, tau_max, n_tau)

real_bc = np.ones(n_tau)

def g_var(c, tau_current):
    f_array = np.zeros(n_grid)
    f_array[0] = -c[0]**2/h**2#*np.interp(tau_current, tau, real_bc)
    return f_array

## Differential operator (varying in space diffusion coefficient case)
Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
vec = np.zeros(n_grid)
vec[0] = 1
Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
Dx /= h # FD derivative matrix

def D_c_var(c):
    Diff =  - Dx.T @ np.diag(c**2) @ Dx
    Diff[0,:] = 0
    Diff[0,0] = c[0]**2/h**2
    return Diff

## PDE form (varying in space diffusion coefficient case)
initial_condition = np.zeros(n_grid)

def PDE_form(c, tau_current):
    return (D_c_var(c), g_var(c, tau_current), initial_condition)

# Create PDE object
PDE = TimeDependentLinearPDE(PDE_form,
                             tau,
                             grid_sol=grid,
                             method='backward_euler',
                             time_obs='all') 

# domain geometry
G_c = MappedGeometry( Continuous1D(grid),  map=lambda x: x)

# range geometry
G_cont2D = Continuous2D((grid, tau))

# Create the CUQIpy PDE model
A = PDEModel(PDE, range_geometry=G_cont2D, domain_geometry=G_c)


#%% 3 - Create Data

# c is a smooth step function
grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)
def smooth_step(x):
    return 10*1/(1+np.exp(-x))
true_c = smooth_step(grid_c_fine-2.5)
plt.figure()
plt.plot(grid_c_fine, true_c)
plt.title('True c')

exact_data = A(CUQIarray(true_c, geometry=G_c, is_par=True))
plt.figure()
exact_data.plot()

#plot 10 solutions
plt.figure()
for i in range(10):
    plt.plot(grid, exact_data.funvals[:,5*i], label='t={}'.format(tau[5*i]))
plt.legend()
    

#%% 4 - Create value of c at which we want to compute the gradient
c_prime =  np.ones(len(grid_c_fine))*4
y_prime = A(CUQIarray(c_prime, geometry=G_c, is_par=True))

#%% 5 - Adjoint problem
real_bc_adj = np.zeros(n_tau)

def g_var_adj(c, tau_current):
    #extract index of tau_current in tau
    index = np.where(np.isclose(tau,tau_current))[0][0]
    print(index)
    f_array = -1/(tau[-1]-tau[-2])*(y_prime.funvals[:,index] - exact_data.funvals[:,index] )
    f_array[0] = 0
    return f_array

## Differential operator (varying in space diffusion coefficient case)

#D_c_var_adj = lambda c:  -Dx.T @ np.diag(c**2) @ Dx # differential operator is symmetric, no need to transpose
def D_c_var_adj(c): 
    Diff =  - Dx.T @ np.diag(c**2) @ Dx
    Diff[0,1:] = 0
    #Diff[1:,0] = 0
    Diff[0,0] = c[0]**2/h**2
    return Diff


## PDE form (varying in space diffusion coefficient case)
final_condition = np.zeros(n_grid)

def PDE_form_adj(c, tau_current):
    return (D_c_var_adj(c), g_var_adj(c, tau_current), final_condition)

# Create PDE object
PDE_adj = TimeDependentLinearPDE(PDE_form_adj,
                             tau,
                             grid_sol=grid,
                             method='backward_euler',
                             time_obs='all')

# c is a smooth step function
PDE_adj.assemble(c_prime)
sol_adj = PDE_adj.solve()

# plot the solution
plt.figure()
for i in range(10):
    plt.plot(grid, sol_adj[0][:,5*i], label='t={}'.format(tau[5*i]))
    plt.title('Adjoint solution')
plt.legend()

# %% 6- Plot 10 source terms
plt.figure()
for i in range(10):
    index=5*i
    rhs = y_prime.funvals[:,index] - exact_data.funvals[:,index] 

    plt.plot(grid, rhs, label='t={}'.format(tau[index]))
    plt.title('Source term')
plt.legend()
# %% 7- Compute gradient


#lambda c:  -Dx.T @ np.diag(c**2) @ Dx 
grad = 0
for i in range(len(tau)):
    u_k = y_prime.funvals[:,index]

    Cu_k = (-Dx.T@np.diag(2*c_prime*(Dx@u_k))).T 
    v_k = sol_adj[0][:,i]
    #grad -= (tau[-1]-tau[-2])*(u_k).T*(sol_adj[0][:,i])
    grad -= (tau[-1]-tau[-2])*Cu_k@v_k
plt.figure()
plt.plot(grid_c_fine, grad)
plt.title('Gradient')




#  8- Compute gradient using scipy
#Objective
def obj(c):
    A_c = A(c)
    return 0.5*(A_c-exact_data).T@(A_c-exact_data)

from scipy.optimize import approx_fprime
grad_scipy = approx_fprime(c_prime, obj, 1e-2)


# plot gradient
#plt.figure()
plt.plot(grid_c_fine, grad_scipy)
plt.title('Gradient scipy')
# %%
