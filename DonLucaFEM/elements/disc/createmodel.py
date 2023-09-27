
# PACKAGES

import numpy as np
import itertools
from scipy.linalg import fractional_matrix_power

if __name__ == '__main__':
    from element import Disc_node, Disc_Element
else:
    from .element import Disc_node, Disc_Element

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


def create_model_raw(XY, elems, E_, nu_, rho_, h_):
    n_elems = np.size(elems, axis=0)
    n_nodes = np.size(XY, axis=0)
    nodes = [Disc_node(i, XY[i]) for i in range(n_nodes)]
    elements = [Disc_Element(elems[j], XY, 
                             E_[j], nu_[j], 
                             rho_[j], h_[j]) 
                for j in range(n_elems)]
    return nodes, elements


def create_K_glob(nodes, elements):
    n_elems = len(elements)
    n_dofs = 2*len(nodes)
    K_glob = np.zeros((n_dofs,n_dofs))
    for j in range(n_elems):
        ind0 = elements[j].nodes_list[0]
        ind1 = elements[j].nodes_list[1]
        ind2 = elements[j].nodes_list[2]
        ind3 = elements[j].nodes_list[3]
        
        # 16 blocks of this matrix
        K_glob[2*ind0:2*ind0+2,2*ind0:2*ind0+2]+=elements[j].K_loc()[0:2,0:2]
        K_glob[2*ind0:2*ind0+2,2*ind1:2*ind1+2]+=elements[j].K_loc()[0:2,2:4]
        K_glob[2*ind0:2*ind0+2,2*ind2:2*ind2+2]+=elements[j].K_loc()[0:2,4:6]
        K_glob[2*ind0:2*ind0+2,2*ind3:2*ind3+2]+=elements[j].K_loc()[0:2,6:8]

        K_glob[2*ind1:2*ind1+2,2*ind0:2*ind0+2]+=elements[j].K_loc()[2:4,0:2]
        K_glob[2*ind1:2*ind1+2,2*ind1:2*ind1+2]+=elements[j].K_loc()[2:4,2:4]
        K_glob[2*ind1:2*ind1+2,2*ind2:2*ind2+2]+=elements[j].K_loc()[2:4,4:6]
        K_glob[2*ind1:2*ind1+2,2*ind3:2*ind3+2]+=elements[j].K_loc()[2:4,6:8]

        K_glob[2*ind2:2*ind2+2,2*ind0:2*ind0+2]+=elements[j].K_loc()[4:6,0:2]
        K_glob[2*ind2:2*ind2+2,2*ind1:2*ind1+2]+=elements[j].K_loc()[4:6,2:4]
        K_glob[2*ind2:2*ind2+2,2*ind2:2*ind2+2]+=elements[j].K_loc()[4:6,4:6]
        K_glob[2*ind2:2*ind2+2,2*ind3:2*ind3+2]+=elements[j].K_loc()[4:6,6:8]
        
        K_glob[2*ind3:2*ind3+2,2*ind0:2*ind0+2]+=elements[j].K_loc()[6:8,0:2]
        K_glob[2*ind3:2*ind3+2,2*ind1:2*ind1+2]+=elements[j].K_loc()[6:8,2:4]
        K_glob[2*ind3:2*ind3+2,2*ind2:2*ind2+2]+=elements[j].K_loc()[6:8,4:6]
        K_glob[2*ind3:2*ind3+2,2*ind3:2*ind3+2]+=elements[j].K_loc()[6:8,6:8]
    return K_glob

def create_M_glob(nodes, elements):
    n_elems = len(elements)
    n_dofs = 2*len(nodes)
    M_glob = np.zeros((n_dofs,n_dofs))
    for j in range(n_elems):
        ind0 = elements[j].nodes_list[0]
        ind1 = elements[j].nodes_list[1]
        ind2 = elements[j].nodes_list[2]
        ind3 = elements[j].nodes_list[3]
        
        # 16 blocks of this matrix
        M_glob[2*ind0:2*ind0+2,2*ind0:2*ind0+2]+=elements[j].M_loc()[0:2,0:2]
        M_glob[2*ind0:2*ind0+2,2*ind1:2*ind1+2]+=elements[j].M_loc()[0:2,2:4]
        M_glob[2*ind0:2*ind0+2,2*ind2:2*ind2+2]+=elements[j].M_loc()[0:2,4:6]
        M_glob[2*ind0:2*ind0+2,2*ind3:2*ind3+2]+=elements[j].M_loc()[0:2,6:8]

        M_glob[2*ind1:2*ind1+2,2*ind0:2*ind0+2]+=elements[j].M_loc()[2:4,0:2]
        M_glob[2*ind1:2*ind1+2,2*ind1:2*ind1+2]+=elements[j].M_loc()[2:4,2:4]
        M_glob[2*ind1:2*ind1+2,2*ind2:2*ind2+2]+=elements[j].M_loc()[2:4,4:6]
        M_glob[2*ind1:2*ind1+2,2*ind3:2*ind3+2]+=elements[j].M_loc()[2:4,6:8]

        M_glob[2*ind2:2*ind2+2,2*ind0:2*ind0+2]+=elements[j].M_loc()[4:6,0:2]
        M_glob[2*ind2:2*ind2+2,2*ind1:2*ind1+2]+=elements[j].M_loc()[4:6,2:4]
        M_glob[2*ind2:2*ind2+2,2*ind2:2*ind2+2]+=elements[j].M_loc()[4:6,4:6]
        M_glob[2*ind2:2*ind2+2,2*ind3:2*ind3+2]+=elements[j].M_loc()[4:6,6:8]
        
        M_glob[2*ind3:2*ind3+2,2*ind0:2*ind0+2]+=elements[j].M_loc()[6:8,0:2]
        M_glob[2*ind3:2*ind3+2,2*ind1:2*ind1+2]+=elements[j].M_loc()[6:8,2:4]
        M_glob[2*ind3:2*ind3+2,2*ind2:2*ind2+2]+=elements[j].M_loc()[6:8,4:6]
        M_glob[2*ind3:2*ind3+2,2*ind3:2*ind3+2]+=elements[j].M_loc()[6:8,6:8]
    return M_glob

def create_C_glob(K_glob, M_glob, factor=0.1):
    """
    needs local stiffness k_ and mass m_ matrices of a beam element
    returns local damping matrix of a beam element
    factor is a scalar
    """
    C_glob = np.real(fractional_matrix_power(np.matmul(M_glob,K_glob), 0.5))*factor
    return C_glob

def Model_Geom_Data(Lx, Ly, num_x, num_y):

    X, Y = np.linspace(0,Lx,num=num_x), np.linspace(0,Ly,num=num_y)
    x, y = np.meshgrid(X,Y)
    XY = np.array([np.ravel(x),np.ravel(y)]).T

    n_nodes = np.int32(num_x*num_y)
    n_elems = np.int32((num_x-1)*(num_y-1))
    n_dofs = 2*n_nodes

    elems = np.empty((0,4),dtype=int)
    for i in range(num_y-1):
        for j in range(num_x-1):
            elems = np.vstack((elems, 
                                np.array([num_x*i+j,  
                                        num_x*i+j+1, 
                                        num_x*(i+1)+j+1,
                                        num_x*(i+1)+j
                                        ])))
    # maybe just return XY, elems
    return X, Y, x, y, XY, elems

def Constraints(XY: np.array, Lx: float, Ly: float, left_edge_bcs: str, right_edge_bcs: str, bottom_edge_bcs: str, top_edge_bcs: str):
    left_edge_nodes_list = np.ravel(np.where((XY[:,0]==0)))
    right_edge_nodes_list = np.ravel(np.where((XY[:,0]==Lx)))
    bottom_edge_nodes_list = np.ravel(np.where((XY[:,1]==0)))
    top_edge_nodes_list = np.ravel(np.where((XY[:,1]==Ly)))

    left_edge_u_x_dofs_list = [2*j for j in left_edge_nodes_list]
    left_edge_u_y_dofs_list = [2*j+1 for j in left_edge_nodes_list]

    right_edge_u_x_dofs_list = [2*j for j in right_edge_nodes_list]
    right_edge_u_y_dofs_list = [2*j+1 for j in right_edge_nodes_list]

    bottom_edge_u_x_dofs_list = [2*j for j in bottom_edge_nodes_list]
    bottom_edge_u_y_dofs_list = [2*j+1 for j in bottom_edge_nodes_list]

    top_edge_u_x_dofs_list = [2*j for j in top_edge_nodes_list]
    top_edge_u_y_dofs_list = [2*j+1 for j in top_edge_nodes_list]

    dofs_constrained = []

    # left edge
    if left_edge_bcs == 'C':
        dofs_constrained.append(left_edge_u_x_dofs_list)
        dofs_constrained.append(left_edge_u_y_dofs_list)
    else:
        if left_edge_bcs == 'F':
            pass

    # right edge
    if right_edge_bcs == 'C':
        dofs_constrained.append(right_edge_u_x_dofs_list)
        dofs_constrained.append(right_edge_u_y_dofs_list)
    else:
        if right_edge_bcs == 'F':
            pass
            
    # bottom edge
    if bottom_edge_bcs == 'C':
        dofs_constrained.append(bottom_edge_u_x_dofs_list)
        dofs_constrained.append(bottom_edge_u_y_dofs_list)
    else:
        if bottom_edge_bcs == 'F':
            pass
            
    # top edge
    if top_edge_bcs == 'C':
        dofs_constrained.append(top_edge_u_x_dofs_list)
        dofs_constrained.append(top_edge_u_y_dofs_list)
    else:
        if top_edge_bcs == 'F':
            pass
    

    dofs_constrained = sorted(list(set(list(itertools.chain.from_iterable(dofs_constrained)))))
    
    return dofs_constrained

def Edge_Dofs(XY: np.array, Lx: float, Ly: float):
        
    '''
    Returns left, right, bottom and top: u_x, then u_y edge dofs lists.
    '''
    left_edge_nodes_list = np.ravel(np.where((XY[:,0]==0)))
    right_edge_nodes_list = np.ravel(np.where((XY[:,0]==Lx)))
    bottom_edge_nodes_list = np.ravel(np.where((XY[:,1]==0)))
    top_edge_nodes_list = np.ravel(np.where((XY[:,1]==Ly)))

    left_edge_u_x_dofs_list = [2*j for j in left_edge_nodes_list]
    left_edge_u_y_dofs_list = [2*j+1 for j in left_edge_nodes_list]

    right_edge_u_x_dofs_list = [2*j for j in right_edge_nodes_list]
    right_edge_u_y_dofs_list = [2*j+1 for j in right_edge_nodes_list]

    bottom_edge_u_x_dofs_list = [2*j for j in bottom_edge_nodes_list]
    bottom_edge_u_y_dofs_list = [2*j+1 for j in bottom_edge_nodes_list]

    top_edge_u_x_dofs_list = [2*j for j in top_edge_nodes_list]
    top_edge_u_y_dofs_list = [2*j+1 for j in top_edge_nodes_list]
    
    return left_edge_u_x_dofs_list, left_edge_u_y_dofs_list,\
        right_edge_u_x_dofs_list, right_edge_u_y_dofs_list,\
            bottom_edge_u_x_dofs_list, bottom_edge_u_y_dofs_list,\
                top_edge_u_x_dofs_list, top_edge_u_y_dofs_list
        


def delete_nodes(XY, elems, deleted_nodes):
    left_nodes = np.delete(np.arange(np.size(XY, axis=0)), deleted_nodes)

    idx_del = []
    idx_left = [j for j in range(np.size(elems, axis=0))]

    for nods in deleted_nodes:
        for j in range(np.size(elems, axis=0)):
            if np.any(elems[j] == nods):
                idx_del.append(j)
                try:
                    idx_left.remove(j)
                except ValueError:
                    pass

    idx_del = np.unique(np.sort(idx_del))

    elems_to_del = elems[idx_del]
    elems_to_leave = elems[idx_left]

    difference = 0*elems_to_leave

    for j in deleted_nodes:
        difference -= 1*(elems_to_leave>=j)

    print(difference)

    elems = elems_to_leave+difference
    XY = XY[left_nodes]
    
    return XY, elems