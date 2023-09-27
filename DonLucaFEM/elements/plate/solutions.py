import numpy as np
import itertools
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh

if __name__ == '__main__':
    from element import Plate_node, Thin_Plate_Element
    from createmodel import *
else:
    from .element import Plate_node, Thin_Plate_Element
    from .createmodel import *

def solve_statics_model_KF(K_glob, f_glob, dofs_constrained):
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

def solve_statics_model_raw(XY, elems, E_, nu_, rho_, h_, dofs_constrained, f_glob):
    '''
    Solves linear static problem for given raw data.
    '''
    nodes, elements = create_model_raw(XY, elems, E_, nu_, rho_, h_)
    K_glob = create_K_glob(nodes, elements)
    U_glob = solve_statics_model_KF(K_glob, f_glob, dofs_constrained)
    
    return nodes, elements, U_glob

def solve_eigenproblem_raw(XY, elems, E_, nu_, rho_, h_, dofs_constrained, max_freq_num, min_freq_num=0, eigvals_only=True):
    '''
    Solves eigenproblem for given raw data.
    '''
    nodes, elements = create_model_raw(XY, elems, E_, nu_, rho_, h_)
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















def solve_statics_full(XY, elems, E_, nu_, rho_, h_, dofs_constrained, f_glob):
    n_elems = np.size(elems, axis=0)
    n_nodes = np.size(XY, axis=0)
    nodes = [Plate_node(i, XY[i]) for i in range(n_nodes)]
    elements = [Thin_Plate_Element(elems[j], XY, E_[j], nu_[j], rho_[j], h_[j]) for j in range(n_elems)]
    n_nodes = len(nodes)
    n_dofs = 3*n_nodes
    dofs_all = np.arange(n_dofs)
    dofs_unconstrained = np.ravel(np.delete(dofs_all,dofs_constrained))
    try:
        F_glob = f_glob
    except NameError:
        F_glob = np.zeros(3*n_nodes)
    K_glob = create_K_glob(nodes, elements)
    K_glob_bc = np.delete(np.delete(K_glob, dofs_constrained, axis=0), dofs_constrained, axis=1)
    F_glob_bc = np.delete(F_glob, dofs_constrained)
    U_glob_bc = np.linalg.solve(K_glob_bc, F_glob_bc)
    U_glob = np.zeros(n_dofs)
    U_glob[dofs_unconstrained] = U_glob_bc
    return U_glob

