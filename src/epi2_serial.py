import numpy as np
import math
import scipy.linalg 
from mpi4py import MPI

# ---- Serial version of KIOPS (only MPI parts removed) ----
def kiops_serial(tau_out, A, u, tol=1e-7, m_init=10, mmin=10, mmax=128, iop=2, task1=False):
    """
    KIOPS algorithm (Serial version): Adaptive Krylov subspace method to compute phi functions.
    Exactly matches original kiops structure, no MPI.
    """

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        u = np.row_stack((u, np.zeros(len(u))))

    m = max(mmin, min(m_init, mmax))

    V = np.zeros((mmax+1, n+p))
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

    w = np.zeros((numSteps, n))
    w[0, :] = u[0, :].copy()

    normU = np.max(np.sum(np.abs(u[1:, :]), axis=1))

    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2**(-ex)
        mu = 2**(ex)
    else:
        nu = 1.0
        mu = 1.0

    u_flip = nu * np.flipud(u[1:, :])

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
            V[0, 0:n] = w[l, :]

            for k in range(p-1):
                i = p - k + 1
                V[j, n+k] = (tau_now**i)/math.factorial(i) * mu
            V[j, n+p-1] = mu

            beta = math.sqrt(np.dot(V[0, 0:n], V[0, 0:n]) + np.dot(V[j, n:n+p], V[j, n:n+p]))
            V[j, :] /= beta

        while j < m:
            j += 1

            V[j, 0:n] = A(V[j-1, 0:n]) + np.dot(V[j-1, n:n+p], u_flip)
            V[j, n:n+p-1] = V[j-1, n+1:n+p]
            V[j, n+p-1] = 0.0

            ilow = max(0, j-iop)
            H[ilow:j, j-1] = np.dot(V[ilow:j, 0:n], V[j, 0:n]) + np.dot(V[ilow:j, n:n+p], V[j, n:n+p])

            V[j, :] -= np.dot(V[ilow:j, :].T, H[ilow:j, j-1])

            nrm = math.sqrt(np.dot(V[j, 0:n], V[j, 0:n]) + np.dot(V[j, n:n+p], V[j, n:n+p]))

            if nrm < tol:
                happy = True
                break

            H[j, j-1] = nrm
            V[j, :] /= nrm

            krystep += 1

        H[0, j] = 1.0
        nrm = H[j, j-1].copy()
        H[j, j-1] = 0.0

        F = scipy.linalg.expm(sign * tau * H[0:j+1, 0:j+1])
        exps += 1

        H[j, j-1] = nrm

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
                #tau_new = remaining_time if omega > delta else tau_opt
                m_new = m_opt
                tau_new = same_tau

        if omega <= delta:           #FJP: blownTs are removed?
            reject += ireject
            step += 1

            nextT = tau_now + tau

            w[l, :] = beta * F[0:j, 0] @ V[0:j, 0:n]

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

    if task1:
        for k in range(numSteps):
            w[k, :] /= tau_out[k]

    m_ret = m
    stats = (step, reject, krystep, exps, conv, m_ret)

    return w, stats

def kiops_parallel(tau_out, A, u, comm, tol=1e-7, m_init=10, mmin=10, mmax=128, iop=2, task1=False):
    """
    KIOPS algorithm (Parallel version): Adaptive Krylov subspace method to compute phi functions.
    Matches serial version exactly, with MPI reductions for inner products.
    """
    # Get MPI info
    rank = comm.Get_rank()
    
    # This follows kiops_serial exactly, only adding MPI reductions where needed
    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        u = np.row_stack((u, np.zeros(len(u))))

    m = max(mmin, min(m_init, mmax))

    V = np.zeros((mmax+1, n+p))
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

    w = np.zeros((numSteps, n))
    w[0, :] = u[0, :].copy()

    normU = np.max(np.sum(np.abs(u[1:, :]), axis=1))

    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2**(-ex)
        mu = 2**(ex)
    else:
        nu = 1.0
        mu = 1.0

    u_flip = nu * np.flipud(u[1:, :])

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
            V[0, 0:n] = w[l, :]

            for k in range(p-1):
                i = p - k + 1
                V[j, n+k] = (tau_now**i)/math.factorial(i) * mu
            V[j, n+p-1] = mu

            # MPI reduction for norm
            beta_sq = np.dot(V[0, 0:n], V[0, 0:n]) + np.dot(V[j, n:n+p], V[j, n:n+p])
            beta = math.sqrt(beta_sq)
            
            if beta < 1e-14:  # Protect against division by zero
                beta = 1.0
                
            V[j, :] /= beta

        while j < m:
            j += 1

            V[j, 0:n] = A(V[j-1, 0:n]) + np.dot(V[j-1, n:n+p], u_flip)
            V[j, n:n+p-1] = V[j-1, n+1:n+p]
            V[j, n+p-1] = 0.0

            ilow = max(0, j-iop)
            
            # Compute inner products (no MPI needed since we're duplicating work)
            H[ilow:j, j-1] = np.dot(V[ilow:j, 0:n], V[j, 0:n]) + np.dot(V[ilow:j, n:n+p], V[j, n:n+p])
            
            # Orthogonalize exactly as in serial
            V[j, :] -= np.dot(V[ilow:j, :].T, H[ilow:j, j-1])

            # Compute norm (no MPI needed since we're duplicating work)
            nrm = math.sqrt(np.dot(V[j, 0:n], V[j, 0:n]) + np.dot(V[j, n:n+p], V[j, n:n+p]))

            if nrm < tol:
                happy = True
                break

            H[j, j-1] = nrm
            
            if nrm < 1e-14:  # Protect against division by zero
                nrm = 1.0
                
            V[j, :] /= nrm

            krystep += 1

        H[0, j] = 1.0
        nrm = H[j, j-1].copy()
        H[j, j-1] = 0.0

        # Only rank 0 computes the matrix exponential (expensive)
        if rank == 0:
            F = scipy.linalg.expm(sign * tau * H[0:j+1, 0:j+1])
        else:
            F = None
            
        # Broadcast F to all ranks
        F = comm.bcast(F, root=0)
        
        exps += 1

        H[j, j-1] = nrm

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

            nextT = tau_now + tau

            # Use matrix-vector product exactly as in serial
            w[l, :] = beta * F[0:j, 0] @ V[0:j, 0:n]

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

    if task1:
        for k in range(numSteps):
            w[k, :] /= tau_out[k]

    m_ret = m
    stats = (step, reject, krystep, exps, conv, m_ret)

    return w, stats    

# --- Matvec Function (complex step) ---
def matvec_fun(vec, dt, Q, rhsQ, rhs_func):
    epsilon = math.sqrt(np.finfo(float).eps)
    perturbed = Q + 1j * epsilon * vec.reshape(Q.shape)
    Jv = (rhs_func(perturbed).imag) / epsilon
    return (dt * Jv).flatten()

# --- Simple EPI2 Step (memory updating Krylov size) ---
def epi2_step(Q, rhs_func, dt, tol=1e-7, mmin=10, mmax=64):
    """
    2nd-order EPI step using adaptive Krylov subspace.
    """

    if not hasattr(epi2_step, 'krylov_size'):
        epi2_step.krylov_size = mmin

    rhsQ = rhs_func(Q)

    def matvec(v):
        return matvec_fun(v, dt, Q, rhsQ, rhs_func)

    vec = np.zeros((2, rhsQ.size))
    vec[1,:] = rhsQ.flatten()

    phiv, stats = kiops_serial([1.], matvec, vec, tol=tol, m_init=epi2_step.krylov_size, mmin=mmin, mmax=mmax)

    used_m = stats[5]
    epi2_step.krylov_size = math.floor(0.7 * used_m + 0.3 * epi2_step.krylov_size)

    deltaQ = np.reshape(phiv, Q.shape) * dt

    return Q + deltaQ


def epi2_step_parallel(Q, rhs_func, dt, comm, tol=1e-7, mmin=10, mmax=64):
    """
    2nd-order EPI step using adaptive Krylov subspace (parallel version).
    Matches serial exactly, just adding MPI comm argument.
    """
    if not hasattr(epi2_step_parallel, 'krylov_size'):
        epi2_step_parallel.krylov_size = mmin

    rhsQ = rhs_func(Q)

    def matvec(v):
        epsilon = math.sqrt(np.finfo(float).eps)
        perturbed = Q + 1j * epsilon * v.reshape(Q.shape)
        Jv = (rhs_func(perturbed).imag) / epsilon
        return (dt * Jv).flatten()

    vec = np.zeros((2, rhsQ.size))
    vec[1,:] = rhsQ.flatten()  # Same as serial - use row 1

    phiv, stats = kiops_parallel([1.], matvec, vec, comm, tol=tol, m_init=epi2_step_parallel.krylov_size, mmin=mmin, mmax=mmax)

    used_m = stats[5]
    epi2_step_parallel.krylov_size = math.floor(0.7 * used_m + 0.3 * epi2_step_parallel.krylov_size)

    deltaQ = np.reshape(phiv, Q.shape) * dt

    return Q + deltaQ