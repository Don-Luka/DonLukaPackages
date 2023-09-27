import numpy as np
import os
import sys
import time

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from tqdm import tqdm, trange
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh


if __name__ == '__main__':
    from element import D2_frame_element, D2_frame_node
    from createmodel import *
else:
    from .element import D2_frame_element, D2_frame_node
    from .createmodel import *

def solve_statics_model_KF(K_glob, f_glob, dofs_constrained):
    '''
    Solves linear static problem for given global stiffness matrix, load vector and boundary conditions.
    '''
    n_dofs = np.size(K_glob, axis=0)
    dofs_all = np.arange(n_dofs)
    dofs_unconstrained = np.ravel(np.delete(dofs_all,dofs_constrained))
    try:
        F_glob = f_glob
    except NameError:
        F_glob = np.zeros(n_dofs)
    K_glob_bc = apply_BC_Matr(K_glob, dofs_constrained)
    F_glob_bc = apply_BC_Vec(F_glob, dofs_constrained)
    U_glob_bc = np.linalg.solve(K_glob_bc, F_glob_bc)
    U_glob = restore_BC(U_glob_bc, n_dofs, dofs_constrained)
    return U_glob

def solve_eigenproblem_KM(K_glob, M_glob, dofs_constrained, max_freq_num, min_freq_num=0, eigvals_only=True):
    '''
    Solves eigenproblem for given global stiffness and mass matrices, and boundary conditions.
    '''
    n_dofs = np.size(K_glob, axis=0)
    dofs_all = np.arange(n_dofs)
    dofs_unconstrained = np.ravel(np.delete(dofs_all,dofs_constrained))
    K_glob_bc = apply_BC_Matr(K_glob, dofs_constrained)
    M_glob_bc = apply_BC_Matr(M_glob, dofs_constrained)
    max_freq_num = min(max_freq_num, np.size(dofs_unconstrained)-1)
    if eigvals_only == True:
        eigenValues = eigh(K_glob_bc, M_glob_bc, eigvals_only=True, subset_by_index=[min_freq_num, max_freq_num])
        return eigenValues
    else:
        eigenValues, eigenVectors = eigh(K_glob_bc, M_glob_bc, eigvals_only=False, subset_by_index=[min_freq_num, max_freq_num])
        U_glob = restore_BC(eigenVectors, n_dofs, dofs_constrained)
        return eigenValues, U_glob

def solve_statics_model_raw(XY, elems, E_, nu_, rho_, k_, b_, h_, dofs_constrained, f_glob):
    '''
    Solves linear static problem for given raw data.
    '''
    nodes, elements = create_model_raw(XY, elems, E_, nu_, rho_, k_, b_, h_)
    K_glob = create_K_glob(nodes, elements)
    U_glob = solve_statics_model_KF(K_glob, f_glob, dofs_constrained)
    
    return nodes, elements, U_glob

def solve_eigenproblem_raw(XY, elems, E_, nu_, rho_, k_, b_, h_, dofs_constrained, max_freq_num, min_freq_num=0, eigvals_only=True):
    '''
    Solves eigenproblem for given raw data.
    '''
    nodes, elements = create_model_raw(XY, elems, E_, nu_, rho_, k_, b_, h_)
    K_glob = create_K_glob(nodes, elements)
    M_glob = create_M_glob(nodes, elements)
    
    return solve_eigenproblem_KM(K_glob, M_glob, dofs_constrained, max_freq_num, min_freq_num, eigvals_only)

def solve_forced_response_once_KMCF(K_glob_bc, M_glob_bc, C_glob_bc, qs_bc, qc_bc, omega):
    """Zwraca wektory odpowiedzi rezonansowej: część cosinusową, sinusową 
    i amplitudę (pierwiastek z sumy kwadratów) ze znakiem dodatnim.
    Uwaga: zwraca tylko część rzeczywistą amplitud."""
    D=(K_glob_bc-M_glob_bc*omega**2)
    a_S = np.matmul(np.linalg.inv(D+omega**2*np.matmul(C_glob_bc, np.matmul(np.linalg.inv(D),C_glob_bc))),(qs_bc+omega*np.matmul(C_glob_bc, np.matmul(np.linalg.inv(D),qc_bc))))
    a_C = np.matmul(np.linalg.inv(D),qc_bc)-omega*np.matmul(np.linalg.inv(D),np.matmul(C_glob_bc, a_S))
    a_CS=np.array((np.sqrt(a_C**2+a_S**2)))
    return a_S, a_C, a_CS

def solve_forced_response_many(K_glob, M_glob, C_glob, qs, qc, omega_vec, dofs_constrained):
    n_omega = np.size(omega_vec)
    n_dofs = np.size(K_glob, axis=0)
    K_glob_bc, M_glob_bc, C_glob_bc, qs_bc, qc_bc =\
        apply_BC_Matr(K_glob, dofs_constrained),\
            apply_BC_Matr(M_glob, dofs_constrained),\
                apply_BC_Matr(C_glob, dofs_constrained),\
                    apply_BC_Vec(qs, dofs_constrained),\
                        apply_BC_Vec(qc, dofs_constrained)
    n_dofs_bc = n_dofs-np.size(dofs_constrained)
    a_S_bc, a_C_bc, a_CS_bc = np.zeros((n_dofs_bc, n_omega)),\
        np.zeros((n_dofs_bc, n_omega)),\
            np.zeros((n_dofs_bc, n_omega))
    for j in range(n_omega):
        a_S_bc[:,j], a_C_bc[:,j], a_CS_bc[:,j] =\
            solve_forced_response_once_KMCF(K_glob_bc, M_glob_bc, C_glob_bc,
                                            qs_bc, qc_bc, omega_vec[j])
    a_S, a_C, a_CS = restore_BC(a_S_bc, n_dofs, dofs_constrained),\
        restore_BC(a_C_bc, n_dofs, dofs_constrained),\
            restore_BC(a_CS_bc, n_dofs, dofs_constrained),
    return a_S, a_C, a_CS

def total_forced_response(nodes, elements, F_glob_s, F_glob_c, dofs_constrained, damping_factor = 0.05, 
                          max_freq_num=2, min_freq_num=0, n_pts = 1001, max_omega_factor = 0.75):
    K_glob = create_K_glob(nodes, elements)
    M_glob = create_M_glob(nodes, elements)
    C_glob = create_C_glob(K_glob, M_glob, factor=damping_factor)
    qs, qc = F_glob_s, F_glob_c
    eigenValues  = solve_eigenproblem_KM(K_glob, M_glob, 
                              dofs_constrained, 
                              max_freq_num, min_freq_num, 
                              eigvals_only=True)
    omega = np.sqrt(eigenValues)
    omega_vec = np.linspace(0, max_omega_factor*omega[max_freq_num], num = n_pts)
    forced_sol = solve_forced_response_many(K_glob, M_glob, C_glob, qs, qc, omega_vec, dofs_constrained)
    a_S, a_C, a_CS = forced_sol
    freq, freq_vec = omega/2/np.pi, omega_vec/2/np.pi
    return freq, freq_vec, a_S, a_C, a_CS


















# RKF for later


def RKF(K_glob_bc, C_glob_bc, M_glob_bc, F_glob_bc, f_min, T_max, n_dofs, dofs_constrained, dofs_unconstrained):
    
    def solve_IVP(uv_0, t_, K_, C_, M_, q_, amp_, freq_):
        '''
        Returns 
        '''
        # load history
        def qval(amp_, t_, freq_):
            return amp_*np.sin(2*np.pi*freq_*t_)
        # dy/dt = f_u
        def f_u(v_):
            return v_
        # dv/dt = f_v
        def f_v(u_, v_, t_, K_, C_, M_, q_):
            return np.linalg.inv(M_)@ (q_* qval(amp_,t_, freq_)-K_@ u_-C_@ v_)
        def deriv(x_, t_, K_, C_, M_, q_):
            u = x_[::2]
            v = x_[1::2]
            dxdt = np.zeros_like(x_)
            dudt = dxdt[::2]
            dvdt = dxdt[1::2]
            dudt[:] = f_u(v[:])
            dvdt[:] = f_v(u[:], v[:], t_, K_, C_, M_, q_)
            return dxdt
        sol0 = odeint(deriv, uv_0, t_, args = (K_, C_, M_, q_))
        return sol0
    
    U_0, V_0 = np.zeros((n_dofs)), np.zeros((n_dofs))
    V_0[-2] = 5e-02
    UV_0 = np.vstack((U_0, V_0)).T
    uv0 = np.array(np.ravel(np.delete(UV_0, dofs_constrained, axis=0)))
    q = F_glob_bc


    n_T = 10

    mode = 'IVP1'

    t = np.linspace(0, n_T*T_max, num=500)
    time_0 = time.time()
    sol0 = solve_IVP(uv0, t, K_glob_bc, C_glob_bc, M_glob_bc, F_glob_bc, 0.1, f_min)
    print(f'ODE solution took {time.time()-time_0} seconds.')
    U_vs_t = np.transpose(sol0[:,::2])
    V_vs_t = np.transpose(sol0[:,1::2])
    
        
        
    U_glob = np.zeros((n_dofs,np.size(t)))
    U_glob[dofs_unconstrained,:] = 50*U_vs_t
