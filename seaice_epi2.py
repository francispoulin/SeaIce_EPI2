#!/usr/bin/env python

import numpy as np 
import scipy as sp
import netCDF4 as nc
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from src.functions import A_fc, A_cf, D_fc, D_cf
from src.functions import Parameters, Grid, Time
from src.functions import meters, km
from src.functions import seconds, minutes, hours, weeks
from src.functions import create_netcdf_file

from src.epi2_serial import epi2_step
from src.epi2_serial import epi2_step_parallel

# --- Define structures for parameters, grid and time
parameters = Parameters(max_Fu = 1e-4, max_Fv = 0e-4)
grid       = Grid(Nx = 100)
time       = Time(dt = 0.05*seconds, tfinal = 0.1*hours, dt_save = 10*seconds)

# --- Define the RHS of the PDEs
def rhs(Q):
    Nx, dx = grid.Nx, grid.dx
    
    P_star    = parameters.P_star
    e         = parameters.e
    C0        = parameters.C0
    Delta_ref = parameters.Delta_ref

    f     = parameters.f
    r     = parameters.r
    rho_i = parameters.rho_i

    u  = Q[0:Nx]
    v  = Q[Nx:2*Nx]
    h  = h0
    A  = A0
    
    Delta = np.sqrt(1.25*A_fc(D_fc(u, dx))**2 + D_cf(v, dx)**2) + parameters.Delta_ref*1e-2 
    Pp    = h*P_star*e**(C0*(A - 1))
    zeta  = A_cf(Pp)/(2*Delta_ref)*np.tanh(Delta_ref/A_cf(Delta))
            
    du = (D_cf((1.25*D_fc(u, dx) - Delta) * zeta, dx))/(rho_i*A_cf(h)) + f*A_cf(v) - r*u + Fu
    dv = (D_fc((D_cf(v, dx) * A_cf(zeta)), dx))/(rho_i*h )             - f*A_fc(u) - r*v + Fv 

    return np.hstack((du, dv))  

# --- Create a netcdf file (only on rank 0)
file_name = 'seaice_uv.nc'
if rank == 0:
    file, u, v = create_netcdf_file(file_name, time, grid)
else:
    file, u, v = None, None, None

# --- Initial Conditions
u0, v0 = np.zeros(grid.Nx), np.zeros(grid.Nx)
h0, A0 = np.ones(grid.Nx), np.ones(grid.Nx)

Q0 = np.hstack((u0, v0)) 
Q = Q0.copy()

# --- Pick forcing
Fu = np.sin(2*np.pi*grid.xf/grid.Lx) * parameters.max_Fu 
Fv = np.cos(2*np.pi*grid.xc/grid.Lx) * parameters.max_Fv 

# --- Begin the time stepping
count = 1
for i in range(time.Nt-1):
    
    # Use the parallel version with MPI
    Q = epi2_step_parallel(Q, rhs, time.dt, comm=comm)

    # --- Save data (only on rank 0)
    if rank == 0 and np.remainder(i, (time.freq_save-1)) == 0:
        print('t = {:6.2f} hours, max_u = {:10.8f}, max_v = {:10.8f}'.format(i*time.dt/hours, 
              np.max(Q[0:grid.Nx]), np.max(Q[grid.Nx:2*grid.Nx])))        
        u[count,:] = Q[0:grid.Nx]
        v[count,:] = Q[grid.Nx:2*grid.Nx]
        count+=1
        
# Close file and create plots only on rank 0
if rank == 0:
    file.close()

    # --- Plot the solution
    ds = nc.Dataset(file_name)

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(20,8))

    fig.suptitle('1D Sea Ice Model with ROS2: h, A fixed')

    uplt = axs[0].pcolormesh(grid.xf/km, time.times_plot/hours, ds['u']) #, vmin=-1.7, vmax=1.7)
    vplt = axs[1].pcolormesh(grid.xc/km, time.times_plot/hours, ds['v']) #, vmin=-1.7, vmax=1.7)

    axs[0].set_title('u')
    axs[1].set_title('v')

    axs[0].set_xlabel('x (km)')
    axs[1].set_xlabel('x (km)')

    axs[0].set_ylabel('time (hours)')
    axs[1].set_ylabel('time (hours)')

    fig.colorbar(uplt, ax=axs[0])
    fig.colorbar(vplt, ax=axs[1])

    fig.savefig('uv_seaice.png', format = 'png', facecolor='white')
    print("Finished! Results saved to", file_name, "and visualization to uv_seaice.png")
