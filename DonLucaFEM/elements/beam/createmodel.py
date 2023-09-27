import numpy as np
import os
import sys
from tqdm import tqdm, trange
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh


if __name__ == '__main__':
    from element import D2_frame_element, D2_frame_node
else:
    from .element import D2_frame_element, D2_frame_node

def apply_BC_Matr(Matr, dofs_constrained):
    Matr_bc = np.delete(np.delete(Matr, dofs_constrained, axis=0), dofs_constrained, axis=1)
    return Matr_bc

def apply_BC_Vec(Vec, dofs_constrained):
    Vec_bc = np.delete(Vec, dofs_constrained, axis=0)
    return Vec_bc

def restore_BC(Matr_bc, n_dofs, dofs_constrained):
    dofs_all = np.arange(n_dofs)
    dofs_unconstrained = np.ravel(np.delete(dofs_all,dofs_constrained))
    if len(np.shape(Matr_bc))>1:
        Matr = np.zeros((n_dofs,np.shape(Matr_bc)[1]))
        Matr[dofs_unconstrained,:] = Matr_bc
    elif len(np.shape(Matr_bc)) == 1:
        Matr = np.zeros(n_dofs)
        Matr[dofs_unconstrained] = Matr_bc
    return Matr

def create_model_raw(XY, elems, E_, nu_, rho_, k_, b_, h_):
    '''
    Creates nodes and elements as objects.
    '''
    n_elems = np.size(elems, axis=0)
    n_nodes = np.size(XY, axis=0)
    
    nodes = [D2_frame_node(j,XY[j]) for j in range(n_nodes)]
    elements = [D2_frame_element(nodes[elems[j][0]], nodes[elems[j][1]], 
                                        E_[j], nu_[j], rho_[j], k_[j], b_[j], h_[j])
                        for j in range(n_elems)]
    return nodes, elements

def create_K_glob(nodes, elements):
    '''
    Creates global stiffness matrix for a system of nodes and elements (as objects)
    '''
    n_elems = len(elements)
    n_dofs = 3*len(nodes)
    K_glob = np.zeros((n_dofs,n_dofs))
    for j in range(n_elems):
        ind1 = elements[j].i_nn.number
        ind2 = elements[j].k_nn.number
        K_glob[3*ind1:3*ind1+3,3*ind1:3*ind1+3]+=elements[j].K_lin_loc_glob()[0:3,0:3]
        K_glob[3*ind1:3*ind1+3,3*ind2:3*ind2+3]+=elements[j].K_lin_loc_glob()[0:3,3:6]
        K_glob[3*ind2:3*ind2+3,3*ind1:3*ind1+3]+=elements[j].K_lin_loc_glob()[3:6,0:3]
        K_glob[3*ind2:3*ind2+3, 3*ind2:3*ind2+3]+=elements[j].K_lin_loc_glob()[3:6,3:6]
    return K_glob

def create_M_glob(nodes, elements):
    '''
    Creates global mass matrix for a system of nodes and elements (as objects)
    '''
    n_elems = len(elements)
    n_dofs = 3*len(nodes)
    M_glob = np.zeros((n_dofs,n_dofs))
    for j in range(n_elems):
        ind1 = elements[j].i_nn.number
        ind2 = elements[j].k_nn.number
        M_glob[3*ind1:3*ind1+3,3*ind1:3*ind1+3]+=elements[j].M_lin_loc_glob()[0:3,0:3]
        M_glob[3*ind1:3*ind1+3,3*ind2:3*ind2+3]+=elements[j].M_lin_loc_glob()[0:3,3:6]
        M_glob[3*ind2:3*ind2+3,3*ind1:3*ind1+3]+=elements[j].M_lin_loc_glob()[3:6,0:3]
        M_glob[3*ind2:3*ind2+3, 3*ind2:3*ind2+3]+=elements[j].M_lin_loc_glob()[3:6,3:6]
    return M_glob

def create_C_glob(K_glob, M_glob, factor=0.1):
    """
    needs local stiffness k_ and mass m_ matrices of a beam element
    returns local damping matrix of a beam element
    factor is a scalar
    """
    C_glob = np.real(fractional_matrix_power(np.matmul(M_glob,K_glob), 0.5))*factor
    return C_glob