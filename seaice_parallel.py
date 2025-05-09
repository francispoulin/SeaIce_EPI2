#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.linalg
import netCDF4 as nc
import matplotlib.pyplot as plt
import math
from mpi4py import MPI
import pdb

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---- Utility constants and functions ----
# Distance units
meters = 1.0
km = 1000.0

# Time units
seconds = 1.0
minutes = 60.0
hours = 60.0 * minutes
days = 24.0 * hours
weeks = 7.0 * days

# Define Parameter, Grid, and Time classes
class Parameters:
    def __init__(self, max_Fu=1e-4, max_Fv=0e-4):
        self.P_star = 27.5e3            # Pa
        self.e = 2                       # []
        self.C0 = 20                     # []
        self.Delta_ref = 1e-11           # []
        self.f = 1.46e-4                 # 1/s, Coriolis parameter
        self.r = 1e-3                    # 1/s, Damping parameter
        self.rho_i = 900                 # kg/m^3
        self.max_Fu = max_Fu
        self.max_Fv = max_Fv

class Grid:
    def __init__(self, Nx=100):
        self.Nx = Nx
        self.Lx = 1000.0 * km
        self.dx = self.Lx / self.Nx
        self.xc = np.arange(self.dx/2.0, self.Lx, self.dx)
        self.xf = np.arange(0, self.Lx, self.dx)

class Time:
    def __init__(self, dt=0.05*seconds, tfinal=0.1*hours, dt_save=10*seconds):
        self.dt = dt
        self.tfinal = tfinal
        self.Nt = int(round(self.tfinal / self.dt)) + 1
        self.dt_save = dt_save
        self.freq_save = int(self.dt_save / self.dt)
        self.times_plot = np.arange(0, self.tfinal+self.dt_save, self.dt_save)

def create_netcdf_file(file_name, time, grid):
    """Create a NetCDF file for output."""
    file = nc.Dataset(file_name, 'w')
    
    # Add dimensions
    file.createDimension('x', grid.Nx)
    file.createDimension('t', len(time.times_plot))
    
    # Add variables
    u = file.createVariable('u', 'f8', ('t', 'x'))
    v = file.createVariable('v', 'f8', ('t', 'x'))
    
    # Return file and variables
    return file, u, v

# ---- Domain Decomposition Functions ----
def decompose_domain(Nx, size):
    """Divide grid points among processors."""
    base = Nx // size
    remainder = Nx % size
    
    counts = [base + (1 if i < remainder else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    
    return counts, displs

def exchange_ghosts(local_array, comm, left_rank, right_rank):
    """Exchange ghost cells with neighboring processors (periodic boundaries)."""
    if local_array.size == 0:  # Empty array check
        return local_array
        
    # Ensure we're working with real arrays
    local_array = np.real(local_array)
        
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Prepare to exchange ghost cells
    if rank == 0:  # Leftmost processor
        # Only add ghost on the right
        extended = np.zeros(local_array.size + 1)
        extended[:-1] = local_array
        
        # Send to right, receive from right
        send_right = local_array[-1]
        recv_right = comm.sendrecv(send_right, dest=right_rank, source=right_rank)
        extended[-1] = recv_right
            
    elif rank == size - 1:  # Rightmost processor
        # Only add ghost on the left
        extended = np.zeros(local_array.size + 1)
        extended[1:] = local_array
        
        # Send to left, receive from left
        send_left = local_array[0]
        recv_left = comm.sendrecv(send_left, dest=left_rank, source=left_rank)
        extended[0] = recv_left
            
    else:  # Interior processor
        # Add ghosts on both sides
        extended = np.zeros(local_array.size + 2)
        extended[1:-1] = local_array
        
        # Send right, receive from left
        send_right = local_array[-1]
        recv_left = comm.sendrecv(send_right, dest=right_rank, source=left_rank)
        extended[0] = recv_left
        
        # Send left, receive from right
        send_left = local_array[0]
        recv_right = comm.sendrecv(send_left, dest=left_rank, source=right_rank)
        extended[-1] = recv_right
            
    return extended

# ---- Parallel Derivative Functions ----
def parallel_D_fc(u_local, dx, comm, counts, displs):
    """Forward difference with parallel ghost exchange and periodic boundaries."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set up periodic boundary conditions
    left_rank = rank - 1 if rank > 0 else size - 1  # Wrap to last processor
    right_rank = rank + 1 if rank < size - 1 else 0  # Wrap to first processor
    
    # Exchange ghost cells
    u_extended = exchange_ghosts(u_local, comm, left_rank, right_rank)
    
    # Compute local derivatives
    du_local = np.zeros_like(u_local)
    
    if rank == 0:  # Leftmost processor
        # One ghost on right (periodic from last processor)
        for i in range(u_local.size):
            next_val = u_extended[-1] if i == u_local.size - 1 else u_local[i+1]
            du_local[i] = (next_val - u_local[i]) / dx
            
    elif rank == size - 1:  # Rightmost processor
        # One ghost on left, and rightmost point wraps to leftmost processor
        for i in range(u_local.size):
            next_val = u_extended[-1] if i == u_local.size - 1 else u_local[i+1]
            du_local[i] = (next_val - u_local[i]) / dx
            
    else:  # Interior processor
        # Ghosts on both sides
        for i in range(u_local.size):
            next_val = u_extended[-1] if i == u_local.size - 1 else u_local[i+1]
            du_local[i] = (next_val - u_local[i]) / dx
    
    return du_local

def parallel_D_cf(u_local, dx, comm, counts, displs):
    """Centered difference with parallel ghost exchange and periodic boundaries."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set up periodic boundary conditions
    left_rank = rank - 1 if rank > 0 else size - 1  # Wrap to last processor
    right_rank = rank + 1 if rank < size - 1 else 0  # Wrap to first processor
    
    # Exchange ghost cells
    u_extended = exchange_ghosts(u_local, comm, left_rank, right_rank)
    
    # Compute local derivatives
    du_local = np.zeros_like(u_local)
    
    if rank == 0:  # Leftmost processor
        for i in range(u_local.size):
            if i == 0:
                # First point uses left ghost (from last processor)
                du_local[i] = (u_local[i] - u_extended[0]) / dx
            else:
                # Other points use centered difference
                du_local[i] = (u_local[i] - u_local[i-1]) / dx
                
    elif rank == size - 1:  # Rightmost processor
        for i in range(u_local.size):
            if i == u_local.size - 1:
                # Last point uses right ghost (from first processor)
                du_local[i] = (u_extended[-1] - u_local[i]) / dx
            else:
                # Other points use centered difference
                du_local[i] = (u_local[i] - u_local[i-1]) / dx
                
    else:  # Interior processor
        for i in range(u_local.size):
            prev_val = u_extended[0] if i == 0 else u_local[i-1]
            du_local[i] = (u_local[i] - prev_val) / dx
    
    return du_local

def parallel_A_fc(u_local, comm, counts, displs):
    """Parallel averaging operation (cell to face) with periodic boundaries."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set up periodic boundary conditions
    left_rank = rank - 1 if rank > 0 else size - 1
    right_rank = rank + 1 if rank < size - 1 else 0
    
    # Exchange ghost cells
    u_extended = exchange_ghosts(u_local, comm, left_rank, right_rank)
    
    # Compute local averages exactly matching serial np.roll(-1) implementation
    au_local = np.zeros_like(u_local)
    
    # For every point, average with the next point (with wraparound for last point)
    for i in range(u_local.size):
        next_idx = i + 1
        
        # If at the last element, use ghost from next processor
        if next_idx >= u_local.size:
            next_val = u_extended[-1]  # Ghost from right (or processor 0 if last proc)
        else:
            next_val = u_local[next_idx]
            
        au_local[i] = 0.5 * (next_val + u_local[i])
    
    return au_local

def parallel_A_cf(u_local, comm, counts, displs):
    """Parallel averaging operation (face to cell) with periodic boundaries."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set up periodic boundary conditions
    left_rank = rank - 1 if rank > 0 else size - 1  # Wrap to last processor
    right_rank = rank + 1 if rank < size - 1 else 0  # Wrap to first processor
    
    # Exchange ghost cells (with periodic wrapping)
    u_extended = exchange_ghosts(u_local, comm, left_rank, right_rank)
    
    # Compute local averages
    au_local = np.zeros_like(u_local)
    
    if rank == 0:  # Leftmost processor 
        for i in range(u_local.size):
            prev_val = u_extended[0] if i == 0 else u_local[i-1]
            au_local[i] = 0.5 * (u_local[i] + prev_val)
    
    elif rank == size - 1:  # Rightmost processor
        for i in range(u_local.size):
            prev_val = u_extended[0] if i == 0 else u_local[i-1]
            au_local[i] = 0.5 * (u_local[i] + prev_val)
    
    else:  # Interior processor
        for i in range(u_local.size):
            idx = i + 1  # Index in extended array with ghost
            prev_val = u_extended[0] if i == 0 else u_extended[idx-1]
            au_local[i] = 0.5 * (u_extended[idx] + prev_val)
    
    return au_local

# ---- Parallel KIOPS Implementation ----
def parallel_kiops(tau_out, A, u_local, comm, tol=1e-7, m_init=10, mmin=10, mmax=128, iop=2, task1=False):
    """
    Parallel KIOPS with domain decomposition.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get local dimensions
    ppo, n_local = u_local.shape
    p = ppo - 1
    
    if p == 0:
        p = 1
        u_local = np.row_stack((u_local, np.zeros(n_local)))
    
    # Get global size via reduction
    n_global = comm.allreduce(n_local, op=MPI.SUM)
    
    m = max(mmin, min(m_init, mmax))
    
    # Create local storage
    V_local = np.zeros((mmax+1, n_local+p))
    H = np.zeros((mmax+1, mmax+1))
    
    step = 0
    krystep = 0
    ireject = 0
    reject = 0
    exps = 0
    sign = np.sign(tau_out[-1])
    tau_now = 0.0
    tau_end = abs(tau_out[-1])
    happy = False
    j = 0
    conv = 0.0
    numSteps = len(tau_out)
    
    w_local = np.zeros((numSteps, n_local))
    w_local[0, :] = u_local[0, :].copy()
    
    # Scale for u_flip
    local_normU = np.max(np.sum(np.abs(u_local[1:, :]), axis=1)) if u_local.shape[0] > 1 else 0
    global_normU = comm.allreduce(local_normU, op=MPI.MAX)
    
    if ppo > 1 and global_normU > 0:
        ex = math.ceil(math.log2(global_normU))
        nu = 2**(-ex)
        mu = 2**(ex)
    else:
        nu = 1.0
        mu = 1.0
    
    u_flip_local = nu * np.flipud(u_local[1:, :]) if u_local.shape[0] > 1 else np.zeros((0, n_local))
    
    tau = tau_end
    
    if tau_end > 1:
        gamma = 0.2
        gamma_mmax = 0.1
    else:
        gamma = 0.9
        gamma_mmax = 0.6
    
    delta = 1.4
    
    oldm = -1
    oldtau = math.nan
    omega = math.nan
    orderold = True
    kestold = True
    l = 0
    
    while tau_now < tau_end:
        if j == 0:
            # Initialize first Krylov vector
            V_local[0, 0:n_local] = w_local[l, :]
            
            # Initialize polynomial terms (same on all processors)
            for k in range(p-1):
                i = p - k + 1
                V_local[j, n_local+k] = (tau_now**i)/math.factorial(i) * mu
            V_local[j, n_local+p-1] = mu
            
            # Compute global norm with reduction
            local_dot1 = np.dot(V_local[0, 0:n_local], V_local[0, 0:n_local])
            local_dot2 = np.dot(V_local[j, n_local:n_local+p], V_local[j, n_local:n_local+p])
            beta = math.sqrt(comm.allreduce(local_dot1 + local_dot2, op=MPI.SUM))
            
            # Protect against division by zero
            if beta < 1e-14:
                beta = 1.0
                
            V_local[j, :] /= beta
        
        while j < m:
            j += 1
            
            # Apply operator A to local part of vector
            V_local[j, 0:n_local] = A(V_local[j-1, 0:n_local])
            
            # Add the polynomial terms contribution
            if u_flip_local.size > 0:
                V_local[j, 0:n_local] += np.dot(V_local[j-1, n_local:n_local+p], u_flip_local)
            
            # Update polynomial terms (same on all processors)
            V_local[j, n_local:n_local+p-1] = V_local[j-1, n_local+1:n_local+p]
            V_local[j, n_local+p-1] = 0.0
            
            ilow = max(0, j-iop)
            
            # Parallel Gram-Schmidt orthogonalization
            for k in range(ilow, j):
                # Compute global inner product using reduction
                local_dot1 = np.dot(V_local[k, 0:n_local], V_local[j, 0:n_local])
                local_dot2 = np.dot(V_local[k, n_local:n_local+p], V_local[j, n_local:n_local+p])
                H[k, j-1] = comm.allreduce(local_dot1 + local_dot2, op=MPI.SUM)
                
                # Local orthogonalization
                V_local[j, :] -= H[k, j-1] * V_local[k, :]
            
            # Compute global norm
            local_norm_sq1 = np.dot(V_local[j, 0:n_local], V_local[j, 0:n_local])
            local_norm_sq2 = np.dot(V_local[j, n_local:n_local+p], V_local[j, n_local:n_local+p])
            nrm = math.sqrt(comm.allreduce(local_norm_sq1 + local_norm_sq2, op=MPI.SUM))
            
            if nrm < tol:
                happy = True
                break
            
            H[j, j-1] = nrm
            
            # Protect against division by zero
            if nrm < 1e-14:
                nrm = 1.0
                
            V_local[j, :] /= nrm
            
            krystep += 1
        
        # Finish building the H matrix
        H[0, j] = 1.0
        nrm = H[j, j-1].copy()
        H[j, j-1] = 0.0
        
        # Only rank 0 computes the matrix exponential, then broadcast
        if rank == 0:
            F = scipy.linalg.expm(sign * tau * H[0:j+1, 0:j+1])
        else:
            F = None
            
        F = comm.bcast(F, root=0)
        exps += 1
        
        H[j, j-1] = nrm
        
        # Adaptive time stepping logic (identical to serial)
        if happy:
            omega = 0.0
            err = 0.0
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new = m
            happy = False
        else: 
            err = abs(beta * nrm * F[j-1, j])
            
            oldomega = omega
            omega = tau_end * err / (tau * tol)
            
            if m == oldm and tau != oldtau and ireject >= 1:
                order = max(1, math.log(omega/oldomega) / math.log(tau/oldtau))
                orderold = False
            elif orderold or ireject == 0:
                orderold = True
                order = j / 4
            else:
                orderold = True
            
            if m != oldm and tau == oldtau and ireject >= 1:
                kest = max(1.1, (omega/oldomega)**(1/(oldm-m)))
                kestold = False
            elif kestold or ireject == 0:
                kest = 2
                kestold = True
            else:
                kestold = True
            
            remaining_time = tau_end - tau_now if omega > delta else tau_end - (tau_now + tau)
            
            same_tau = min(remaining_time, tau)
            tau_opt = tau * (gamma/omega)**(1/order)
            tau_opt = min(remaining_time, max(tau/5, min(5*tau, tau_opt)))
            
            m_opt = math.ceil(j + math.log(omega/gamma) / math.log(kest))
            m_opt = max(mmin, min(mmax, max(math.floor(3/4*m), min(m_opt, math.ceil(4/3*m)))))
            
            if j == mmax:
                if omega > delta:
                    m_new = j
                    tau_new = tau * (gamma_mmax/omega)**(1/order)
                    tau_new = min(tau_end - tau_now, max(tau/5, tau_new))
                else:
                    tau_new = tau_opt
                    m_new = m
            else:
                m_new = m_opt
                tau_new = same_tau
        
        if omega <= delta:
            reject += ireject
            step += 1
            
            # Compute local part of the result
            w_local[l, :] = np.zeros_like(w_local[l, :])
            for k in range(j+1):
                w_local[l, :] += beta * F[k, 0] * V_local[k, 0:n_local]
            
            tau_now += tau
            j = 0
            ireject = 0
            conv += err
        else:
            ireject += 1
            H[0, j] = 0.0
        
        oldtau = tau
        tau = tau_new
        
        oldm = m
        m = m_new
    
    # Scale result if task1 is True (phi_1 function)
    if task1:
        for k in range(numSteps):
            w_local[k, :] /= tau_out[k]
    
    m_ret = m
    stats = (step, reject, krystep, exps, conv, m_ret)
    
    return w_local, stats

# ---- Parallel EPI2 Step ----
def parallel_epi2_step(Q_local, rhs_func_local, dt, comm, counts, displs, tol=1e-7, mmin=10, mmax=64):
    """
    Parallel EPI2 step with domain decomposition.
    """
    if not hasattr(parallel_epi2_step, 'krylov_size'):
        parallel_epi2_step.krylov_size = mmin
    
    # Ensure Q_local is real
    Q_local = np.real(Q_local)  # <-- Add this conversion
    
    # Compute RHS on local portion
    rhsQ_local = rhs_func_local(Q_local, comm, counts, displs)
    
    # Define parallel matvec function
    def parallel_matvec(v_local):
        epsilon = math.sqrt(np.finfo(float).eps)
        
        # Reconstruct perturbed Q on local domain
        perturbed_local = Q_local + 1j * epsilon * v_local.reshape(Q_local.shape)
        
        # Apply RHS function with perturbation
        Jv_local = (rhs_func_local(perturbed_local, comm, counts, displs).imag) / epsilon
        
        return (dt * Jv_local).flatten()
    
    # Set up vectors for parallel KIOPS
    vec_local = np.zeros((2, rhsQ_local.size))
    vec_local[1, :] = rhsQ_local.flatten()
    
    # Call parallel KIOPS
    phiv_local, stats = parallel_kiops([1.], parallel_matvec, vec_local, comm, 
                                     tol=tol, m_init=parallel_epi2_step.krylov_size, 
                                     mmin=mmin, mmax=mmax)
    
    # Update Krylov size based on statistics
    used_m = stats[5]
    parallel_epi2_step.krylov_size = math.floor(0.7 * used_m + 0.3 * parallel_epi2_step.krylov_size)
    
    # Update local solution
    deltaQ_local = np.reshape(phiv_local, Q_local.shape) * dt
    
    # Return real result
    return np.real(Q_local + deltaQ_local)  # <-- Add np.real here

# ---- Define parallel RHS function ----
def parallel_rhs(Q_local, comm, counts, displs):
    """Parallel version of RHS function with domain decomposition."""
    # Ensure Q_local is real if we might be getting complex values
    Q_local = np.real(Q_local)  # <-- Add this conversion

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Extract local grid size
    Nx_local = counts[rank]
    
    # Get global parameters from serial code
    P_star = parameters.P_star
    e = parameters.e
    C0 = parameters.C0
    Delta_ref = parameters.Delta_ref
    f = parameters.f
    r = parameters.r
    rho_i = parameters.rho_i
    dx = grid.dx
    
    # Split local Q into u and v
    u_local = Q_local[0:Nx_local]
    v_local = Q_local[Nx_local:2*Nx_local]
    
    # Get global h0 and A0
    h_local = h0_local
    A_local = A0_local
    
    # Compute derivatives using parallel functions
    Du_local = parallel_D_fc(u_local, dx, comm, counts, displs)
    Dv_local = parallel_D_cf(v_local, dx, comm, counts, displs)
    
    # Compute Delta using parallel operations
    AFc_Du_local = parallel_A_fc(Du_local, comm, counts, displs)
    Delta_local = np.sqrt(1.25*AFc_Du_local**2 + Dv_local**2) + parameters.Delta_ref*1e-2
    
    # Compute pressure
    Pp_local = h_local*P_star*np.exp(C0*(A_local - 1))
    
    # Compute zeta
    ACf_Pp_local = parallel_A_cf(Pp_local, comm, counts, displs)
    ACf_Delta_local = parallel_A_cf(Delta_local, comm, counts, displs)
    zeta_local = ACf_Pp_local/(2*Delta_ref)*np.tanh(Delta_ref/ACf_Delta_local)
    
    # Compute du and dv
    term1 = (1.25*Du_local - Delta_local) * zeta_local
    DCf_term1 = parallel_D_cf(term1, dx, comm, counts, displs)
    ACf_h_local = parallel_A_cf(h_local, comm, counts, displs)
    ACf_v_local = parallel_A_cf(v_local, comm, counts, displs)
    du_local = DCf_term1/(rho_i*ACf_h_local) + f*ACf_v_local - r*u_local + Fu_local
    
    ACf_zeta_local = parallel_A_cf(zeta_local, comm, counts, displs)
    term2 = Dv_local * ACf_zeta_local
    DFc_term2 = parallel_D_fc(term2, dx, comm, counts, displs)
    AFc_u_local = parallel_A_fc(u_local, comm, counts, displs)
    dv_local = DFc_term2/(rho_i*h_local) - f*AFc_u_local - r*v_local + Fv_local
    
    # Combine results
    return np.hstack((du_local, dv_local))

def serial_rhs(Q):
    from src.functions import A_fc, A_cf, D_fc, D_cf
    from src.functions import Parameters, Grid, Time
    Nx, dx = grid.Nx, grid.dx
    
    P_star    = parameters.P_star
    e         = parameters.e
    C0        = parameters.C0
    Delta_ref = parameters.Delta_ref

    f     = parameters.f
    r     = parameters.r
    rho_i = parameters.rho_i
    
    h0, A0 = np.ones(grid.Nx), np.ones(grid.Nx)
    u  = Q[0:Nx]
    v  = Q[Nx:2*Nx]
    h  = h0
    A  = A0
    pdb.set_trace()
    Delta = np.sqrt(1.25*A_fc(D_fc(u, dx))**2 + D_cf(v, dx)**2) + parameters.Delta_ref*1e-2 
    Pp    = h*P_star*e**(C0*(A - 1))
    zeta  = A_cf(Pp)/(2*Delta_ref)*np.tanh(Delta_ref/A_cf(Delta))

    # --- Pick forcing
    Fu = np.sin(2*np.pi*grid.xf/grid.Lx) * parameters.max_Fu 
    Fv = np.cos(2*np.pi*grid.xc/grid.Lx) * parameters.max_Fv 
            
    du = (D_cf((1.25*D_fc(u, dx) - Delta) * zeta, dx))/(rho_i*A_cf(h)) + f*A_cf(v) - r*u + Fu
    dv = (D_fc((D_cf(v, dx) * A_cf(zeta)), dx))/(rho_i*h )             - f*A_fc(u) - r*v + Fv 

    return np.hstack((du, dv))  

def verify_rhs_function():
    """Verify that parallel RHS matches serial with 1 processor."""
    if size != 1 or rank != 0:
        return
    
    # Create test solution
    test_Q = np.hstack((np.sin(np.linspace(0, 2*np.pi, grid.Nx)), 
                        np.cos(np.linspace(0, 2*np.pi, grid.Nx))))
    
    # Compute with both functions
    serial_result = serial_rhs(test_Q)
    parallel_result = parallel_rhs(test_Q, comm, counts, displs)
    
    # Compare results
    max_diff = np.max(np.abs(serial_result - parallel_result))
    print(f"RHS function maximum difference: {max_diff}")

# ---- Main Program ----
# Define structures for parameters, grid and time
parameters = Parameters(max_Fu = 1e-4, max_Fv = 0e-4)
grid = Grid(Nx = 100)
time = Time(dt = 0.05*seconds, tfinal = 0.1*hours, dt_save = 10*seconds)

# Decompose domain
counts, displs = decompose_domain(grid.Nx, size)
Nx_local = counts[rank]

# Create local grid
local_indices = np.arange(displs[rank], displs[rank] + counts[rank])
xc_local = grid.xc[local_indices] if Nx_local > 0 else np.array([])
xf_local = grid.xf[local_indices] if Nx_local > 0 else np.array([])

# Initial Conditions - local portions
u0_local = np.zeros(Nx_local)
v0_local = np.zeros(Nx_local)
h0_local = np.ones(Nx_local)
A0_local = np.ones(Nx_local)

# Local forcing terms
Fu_local = np.sin(2*np.pi*xf_local/grid.Lx) * parameters.max_Fu if Nx_local > 0 else np.array([])
Fv_local = np.cos(2*np.pi*xc_local/grid.Lx) * parameters.max_Fv if Nx_local > 0 else np.array([])

# Combined local solution vector
Q0_local = np.hstack((u0_local, v0_local))
Q_local = Q0_local.copy()

verify_rhs_function()

# Create output file (rank 0 only)
if rank == 0:
    file_name = 'seaice_uv_parallel.nc'
    file, u_global, v_global = create_netcdf_file(file_name, time, grid)
else:
    file, u_global, v_global = None, None, None

# Time stepping loop
count = 1
for i in range(time.Nt-1):
    # Take parallel EPI2 step
    Q_local = parallel_epi2_step(Q_local, parallel_rhs, time.dt, comm, counts, displs)
    
    # Save data (gather to rank 0) - FIXED SAVE CONDITION
    if np.remainder(i, time.freq_save) == 0:  # <-- REMOVED the -1
        # Gather u and v to rank 0
        u_local = Q_local[:Nx_local]
        v_local = Q_local[Nx_local:2*Nx_local]
        
        u_gathered = None
        v_gathered = None
        
        if rank == 0:
            u_gathered = np.zeros(grid.Nx)
            v_gathered = np.zeros(grid.Nx)
        
        # Gather data to rank 0
        comm.Gatherv(u_local, [u_gathered, counts, displs, MPI.DOUBLE], root=0)
        comm.Gatherv(v_local, [v_gathered, counts, displs, MPI.DOUBLE], root=0)
        
        # Rank 0 saves to file (with bounds check)
        if rank == 0:
            print('t = {:6.2f} hours, max_u = {:10.8f}, max_v = {:10.8f}'.format(i*time.dt/hours, 
                  np.max(u_gathered), np.max(v_gathered)))
            
            # Add safety bounds check
            if count < len(time.times_plot):
                u_global[count,:] = u_gathered
                v_global[count,:] = v_gathered
                count += 1
            else:
                print(f"WARNING: Exceeded allocated time dimension (count={count}, dim size={len(time.times_plot)})")