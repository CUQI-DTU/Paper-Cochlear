#%%
import numpy as np
import matplotlib.pyplot as plt
from my_utils import plot_time_series
import pandas as pd
from cuqi.distribution import Gaussian, JointDistribution, GMRF
from cuqi.geometry import Continuous1D, KLExpansion, Discrete, MappedGeometry, Continuous2D, Image2D
from cuqi.pde import TimeDependentLinearPDE
from cuqi.model import PDEModel, Model
from cuqi.samples import Samples 
import os

earlist = ['l']
animallist = ['m1']
#version_list = ['', 'v2', 'v3', 'v4']
#version_list_labels = ['CWMH_10000', 'CWMH_50000', 'NUTS_1000', 'MH_1000000']
version_list = ['v_aug2_c', 'v_aug2_c']


# Do not loop over the following lists
sampler_list = ['NUTS', 'NUTS']
unknown_par_type_list = ['constant', 'smooth']
unknown_par_value_list = [[100.0], [400.0, 1200.0]]
data_type_list = ['synthetic', 'synthetic']
inference_type_list = ['both', 'both']
Ns_list = [[1000, 1000], [1000, 1000]]
noise_level_list = [0.1, 0.1]

#version_list_labels = ['']

global_expr_id = -1

for i, version in enumerate(version_list):
    
    for ear in earlist:
        for animal in animallist:
            global_expr_id += 1
            tag = ''
            upvl = unknown_par_value_list[global_expr_id]
            if len(upvl) == 1:
                unknown_par_value_str = str(upvl[0])
            elif len(upvl) == 2:
                unknown_par_value_str = str(upvl[0])+\
                    '_'+str(upvl[1])
            else:
                raise Exception('Unknown parameter value not supported')
            
            ## Create directory for output
            tag = animal+ear+sampler_list[global_expr_id]\
                +unknown_par_type_list[global_expr_id]\
                +unknown_par_value_str\
                +data_type_list[global_expr_id]\
                +inference_type_list[global_expr_id]\
                +str(Ns_list[global_expr_id][0])\
                +str(Ns_list[global_expr_id][1])\
                +str(noise_level_list[global_expr_id])\
                +version

            # Create directory in figures for output and raise an error if it already exists
            out_dir_name = './figures/'+tag
            if not os.path.exists(out_dir_name):
                os.makedirs(out_dir_name)
            else:
                raise Exception('Output directory already exists')
    

            # If the data is not provided, we create it
            if data_type_list[global_expr_id] == 'synthetic':
                # if the unknown parameter is constant
                if unknown_par_type_list[global_expr_id] == 'constant':
                    exact_x = unknown_par_value_list[global_expr_id][0]
                    x_geom = G_D_const
                    exact_data = A_const(unknown_par_value_list[global_expr_id][0])
                
            
                # if the unknown parameter is varying in space (smooth function)
                elif data_type_list[global_expr_id] == 'smooth':
                    low = unknown_par_value_list[global_expr_id][0]
                    high = unknown_par_value_list[global_expr_id][1]
                    exact_x = (high-low)*np.sin(2*np.pi*((L-grid_c))/(4*L)) + low
                    x_geom = G_D_var
                    exact_data = A_var(exact_x)
                exact_x = CUQIarray(exact_x, geometry=x_geom, is_par=False)
    


            #tag = animal+ear+sampler_list[i]+version
            print(tag)
            ## Read data
            ## Read distance file
            dist_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_distances.csv')
            locations = dist_file['distance microns'].values[:5]
            print(locations)
            
            ## Read concentration file
            constr_file = pd.read_csv('../../data/parsed/CT/20210120_'+animal+'_'+ear+'_parsed.csv')
            data = constr_file[['CA1', 'CA2', 'CA3', 'CA4', 'CA5']].values.T.ravel()
            print(data)
            times = constr_file['time'].values*60
            print(times)

            ## Build model:
            ## Set PDE parameters
            L = 500
            n_grid = 50 if (version == 'v3' or version == 'v4') else 100   # Number of solution nodes
            h = L/(n_grid+1)   # Space step size
            grid = np.linspace(h, L-h, n_grid)
            
            tau_max = 30*60 # Final time in sec
            cfl = 5 # The cfl condition to have a stable solution
                     # the method is implicit, we can choose relatively large time steps 
            dt_approx = cfl*h**2 # Defining approximate time step size
            n_tau = int(tau_max/dt_approx)+1 # Number of time steps
            tau = np.linspace(0, tau_max, n_tau)

            ## Initial condition
            initial_condition = np.zeros(n_grid) 
            
            ## Range geometry
            G_cont2D = Continuous2D((locations, times))
            
            ## Noise standard deviation 
            s_noise = 0.1 \
                      *np.linalg.norm(data) \
                      *np.sqrt(1/G_cont2D.par_dim) 
            
            ## Source term (constant diffusion coefficient case)
            def g_const(c, tau_current):
                f_array = np.zeros(n_grid)
                f_array[0] = c/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
                return f_array
            
            ## Differential operator (constant diffusion coefficient case)
            D_c_const = lambda c: c * ( np.diag(-2*np.ones(n_grid), 0) +
            np.diag(np.ones(n_grid-1), -1) +
            np.diag(np.ones(n_grid-1), 1) ) / h**2
             
            ## PDE form (constant diffusion coefficient case)
            def PDE_form_const(c, tau_current):
                return (D_c_const(c), g_const(c, tau_current), initial_condition)
            
            ## CUQIpy PDE object (constant diffusion coefficient case)
            PDE_const = TimeDependentLinearPDE(PDE_form_const, tau, grid_sol=grid,
                                         method='backward_euler', 
                                         grid_obs=locations,
                                        time_obs=times) 
            
            ## Domain geometry (constant diffusion coefficient case)
            G_D_const =  MappedGeometry( Discrete(1),  map=lambda x: x**2 )
            
            ## PDE forward model (constant diffusion coefficient case)
            A_const = PDEModel(PDE_const, range_geometry=G_cont2D, domain_geometry=G_D_const)
            
            # grid for the diffusion coefficient
            n_grid_c = 20
            hs = L/(n_grid_c+1) 
            grid_c = np.linspace(0, L, n_grid_c+1, endpoint=True)
            grid_c_fine = np.linspace(0, L, n_grid+1, endpoint=True)
            
            ## Source term (varying in space diffusion coefficient case)
            def g_var(c, tau_current):
                f_array = np.zeros(n_grid)
                f_array[0] = c[0]/h**2*np.interp(tau_current, times, data.reshape([len(locations), len(times)])[0,:])
                return f_array
            
            ## Differential operator (varying in space diffusion coefficient case)
            Dx = - np.diag(np.ones(n_grid), 0)+ np.diag(np.ones(n_grid-1), 1) 
            vec = np.zeros(n_grid)
            vec[0] = 1
            Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
            Dx /= h # FD derivative matrix
            
            D_c_var = lambda c: - Dx.T @ np.diag(c) @ Dx
            
            def PDE_form_var(c, tau_current):
                c = np.interp(grid_c_fine, grid_c, c)
                return (D_c_var(c), g_var(c, tau_current), initial_condition)
            
            
            PDE_var = TimeDependentLinearPDE(PDE_form_var, tau, grid_sol=grid,
                                         method='backward_euler', 
                                         grid_obs=locations,
                                        time_obs=times) 
            
            
            # Domain geometry
            G_D_var =  MappedGeometry( Continuous1D(grid_c),  map=lambda x: x**2 )
            
            
            A_var = PDEModel(PDE_var, range_geometry=G_cont2D, domain_geometry=G_D_var)


            ## read samples
            dir_name = '../../../Collab-BrainEfflux-Data/ear_aqueducts/output'+tag
            #dir_name = 'output'+tag
            

            try:
                samples_const = Samples(np.load(dir_name+'/posterior_samples_const_'+tag+'.npz')['arr_0'],
                              geometry=G_D_const)
            except:
                continue

            try:
                samples_var = Samples(np.load(dir_name+'/posterior_samples_var_'+tag+'.npz')['arr_0'],
                              geometry=G_D_var)
            except:
                continue

            samples_const = samples_const.burnthin(int(samples_const.Ns*0.3))
            samples_var = samples_var.burnthin(int(samples_var.Ns*0.3))
            
            fig = plt.figure( figsize=(12, 14))
            #fig.title = tag+version_list_labels[i]
            fig.suptitle(tag)
            #fig.set_label(tag+version_list_labels[i])
            subfigs = fig.subfigures(2, 1)
            
            axsTop = subfigs[0].subplots(3, 2)
            axsBottom = subfigs[1].subplots(3, 2)
            
            plt.sca(axsTop[0,0])
            plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )
            plt.title('Data, case '+tag)
            
            plt.sca(axsTop[0,1])
            recon_data_const = A_const(samples_const.funvals.mean(), is_par=False).reshape([len(locations), len(times)])
            plot_time_series( times, locations, recon_data_const)
            plt.title('Recon from constant c')
            
            plt.sca(axsTop[1,0])
            recon_data_var = A_var(samples_var.funvals.mean(), is_par=False).reshape([len(locations), len(times)])
            plot_time_series( times, locations, recon_data_var)
            plt.title('Recon from varying c')
            
            plt.sca(axsTop[1,1])
            samples_const.funvals.plot_ci()
            plt.title('Constant c')
            plt.title('Posterior samples ci (constant c)\n ESS = '+str(samples_const.compute_ess()))
            
            plt.sca(axsTop[2,0])
            samples_var.funvals.plot_ci()
            plt.title('Varying c')
            
            plt.sca(axsTop[2,1])
            plt.plot(samples_var.compute_ess())
            plt.title('ESS for varying c')
            
            
            #plt.sca(axsBottom)
            samples_var.plot_trace([0, 5, 15] ,axes = axsBottom)
            
            fig.savefig(out_dir_name+'/results_'+tag+'.png')

            #fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            #
            #plt.sca(ax[0])
            #plot_time_series( times, locations, data.reshape([len(locations), len(times)]) )
            #
            #plt.sca(ax[1])
            #recon_data = A_var(samples_var.funvals.mean(), is_par=False).reshape([len(locations), len(times)])
            #plot_time_series( times, locations, recon_data)

            ### save figure
            #plt.savefig(dir_name+'/data_recon_var_'+tag+'corrected.png')
