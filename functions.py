#!/usr/bin/env python
import numpy as np

from dataclasses import dataclass, asdict, field

meters = 1
km     = 1e3*meters

seconds = 1
minutes = 60
hours   = 60*minutes
days    = 24*hours
weeks   = 7*days

@dataclass
class Parameters:
    """Class for storing parameters."""
    C0:        float = 20.
    e:         float = 2.
    lne:       float = np.log(e)
    rho_i:     float = 917.
    P_star:    float = 2.7e4
    Delta_ref: float = 2e-9
    f:         float = 7e-5
    r:         float = 0.
    max_Fu:    float = 1e-4
    max_Fv:    float = 1e-4
    
@dataclass
class Grid:
    """Class for storing the grid parameters."""
    Lx:        float = 100*km
    Nx:        int   = 50
    dx:        float = Lx/Nx
    
    xf:        float = field(init=False)
    xc:        float = field(init=False)
    
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        self.xf = np.linspace(0, self.Lx - self.dx, self.Nx)
        self.xc = self.xf + self.dx/2 
        
@dataclass
class Time:
    """Class for storing the time parameters."""
    dt:         float = 10*seconds
    tfinal:     float = 2*weeks
    Nt:         float = field(init=False)

    dt_save:    float = 4*hours      
    freq_save:  int   = field(init=False)     
    num_plots : int   = field(init=False)

    times:      float = field(init=False)
    times_plot: float = field(init=False)
    
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        self.Nt         = int(self.tfinal/self.dt) + 1
        self.freq_save  = int(self.dt_save/self.dt) + 1
        self.num_plots  = int(self.tfinal/self.dt_save) + 1
        self.times      = np.arange(self.Nt)*self.dt 
        self.times_plot = np.arange(self.num_plots)*self.dt_save
        
@dataclass
class Solution:
    """Class for storing the solution."""
    u:         float = field(init=False)        # face
    v:         float = field(init=False)        # center
    h:         float = field(init=False)        # center
    A:         float = field(init=False)        # center
    Delta:     float = field(init=False)        # center
    zeta:      float = field(init=False)        # center

def compute_fluxes(solution, parameters, grid, Fu, Fv):
    u     = solution.u
    v     = solution.v
    h     = solution.h
    zeta  = solution.zeta
    Delta = solution.Delta
    rheology_factor = solution.rheology_factor
    
    dx = grid.dx
    
    f  = parameters.f
    r  = parameters.r
    
    rho_i = parameters.rho_i
    
    du = (D_cf((1.25*D_fc(u, dx) - Delta) * zeta, dx))*rheology_factor/(rho_i*A_cf(h)) + f*A_cf(v) - r*u + Fu
    dv = (D_fc((D_cf(v, dx) * A_cf(zeta)), dx))*rheology_factor/(rho_i*h )             - f*A_fc(u) - r*v + Fv 

    return np.hstack((du, dv))  


def A_fc(u):
    return 0.5*(np.roll(u,-1) + u)

def A_cf(v):
    return 0.5*(v + np.roll(v,1))

def D_fc(u, dx):
    return (np.roll(u,-1) - u)/dx

def D_cf(v, dx):
    return (v - np.roll(v,1))/dx

def plot_solution(solution, grid, parameters, t):
    
    fig, axs = plt.subplots(3, 2, figsize=(20,12))

    fig.suptitle('Profiles at t = ' + '{:.2f}'.format(t/weeks) + ' weeks')

    uplt = axs[0,0].plot(grid.xf/km, solution.u,       '-b',  lw=3, label='u_f')

    vplt = axs[0,1].plot(grid.xc/km, solution.v,        '-b', lw=3, label='v_c')

    Deltplt = axs[1,0].plot(grid.xc/km, solution.Delta,       '-b',  lw=3, label='Delta_c')

    zetaplt = axs[1,1].plot(grid.xf/km, solution.zeta,       '-b',  lw=3, label='zeta_f')

    Ppplt  = axs[2,0].plot(grid.xc/km, solution.Pp,       '-b',  lw=3, label='Pp_c')

    Dozplt = axs[2,1].plot(grid.xf/km, (solution.zeta/solution.Delta),     '-b',  lw=3, label='zeta/D_f')

    axs[0,0].set_title('u')
    axs[0,1].set_title('v')
    axs[1,0].set_title('Delta')
    axs[1,1].set_title('zeta')
    axs[2,0].set_title('Pp')
    axs[2,1].set_title('zeta/Delta')

    axs[2,1].set_xlabel('x (km)')
    axs[2,1].set_xlabel('x (km)')

    axs[0,0].set_xlim([0, grid.Lx/km])
    axs[0,1].set_xlim([0, grid.Lx/km])
    axs[1,0].set_xlim([0, grid.Lx/km])
    axs[1,1].set_xlim([0, grid.Lx/km])
    axs[2,0].set_xlim([0, grid.Lx/km])
    axs[2,1].set_xlim([0, grid.Lx/km])

    axs[0,0].grid()
    axs[0,1].grid()
    axs[1,0].grid()
    axs[1,1].grid()
    axs[2,0].grid()
    axs[2,1].grid()

    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    axs[2,0].legend()
    axs[2,1].legend()    
