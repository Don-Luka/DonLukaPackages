import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def get_auxeticgrid_elems(n_x, n_y):
    '''
    Creates connectivity matrix (elements to nodes)
    '''
    n_nodes = 2*n_x+(4*n_x+2)*n_y
    nodes_list = np.arange(n_nodes)
    elems = []
    # 1st elements layer | |
    for j in range(n_x):
        elems.append([j, n_x+(2*j+1)])

    # 2nd elements layer /\/\
    for j in range(2*n_x):
        elems.append([j+n_x, j+n_x+1])

    # 3rd elements layer | | |
    for j in range(n_x+1):
        elems.append([2*j+n_x, 2*j+n_x+2*n_x+1])

    # 4th elements layer \/\/
    for j in range(2*n_x):
        elems.append([3*n_x+1+j, 3*n_x+2+j])

    elems = np.array(elems)

    # 5th elements layer | |
    for j in range(n_x):
        elems=np.concatenate((elems, np.array([elems[j]+np.array([3*n_x+2+j,4*n_x+2])])))

    for j in range(n_y-1):
        elems=np.concatenate((elems, elems[-(6*n_x+1):]+4*n_x+2))

    elems = elems[:-n_x]

    elems=np.concatenate((elems, np.transpose(np.array([nodes_list[-n_x-(2*n_x+1)+1:-n_x:2],nodes_list[-n_x:]]))))
    
    return elems


def get_auxeticgrid_coords(L_x, L_y, n_x, n_y, a_dim, b_dim, c_dim, d_dim):
    '''
    Creates nodes and their coordinates
    '''
    
    def rescale_cells(L_x, L_y, n_x, n_y, a_dim, b_dim, c_dim, d_dim):
        cell_diam = ((L_x/n_x)**2+(L_y/n_y)**2)**0.5 # may be needed later
        vertical_scale_factor = 1/(((a_dim+b_dim-2*d_dim)*n_y+a_dim)/L_y)
        horizontal_scale_factor = L_x/n_x
        a_dim, b_dim, d_dim = a_dim*vertical_scale_factor, b_dim*vertical_scale_factor, d_dim*vertical_scale_factor
        c_dim = c_dim*horizontal_scale_factor
        return a_dim, b_dim, c_dim, d_dim
    
    a_dim, b_dim, c_dim, d_dim = rescale_cells(L_x, L_y, n_x, n_y, a_dim, b_dim, c_dim, d_dim)
    
    # nodes of all except the last layer
    nodes = []
    for j in range(n_x):
        nodes.append([(2*(j)+1)*c_dim,0])

    for k in range(n_y):

        for j in range(2*n_x+1):
            if (j % 2) == 0:
                nodes.append([j*c_dim,a_dim-d_dim+k*(a_dim+b_dim-2*d_dim)])
            else:
                nodes.append([j*c_dim,a_dim+k*(a_dim+b_dim-2*d_dim)])
                
        for j in range(2*n_x+1):
            if (j % 2) == 0:
                nodes.append([j*c_dim,a_dim+b_dim-d_dim+k*(a_dim+b_dim-2*d_dim)])
            else:
                nodes.append([j*c_dim,a_dim+b_dim-2*d_dim+k*(a_dim+b_dim-2*d_dim)])

    nodes = np.array(nodes)

    for j in range(n_x):
        nodes = np.append(nodes,np.array([[(2*(j)+1)*c_dim, (a_dim+b_dim-2*d_dim)*n_y+a_dim]]),axis=0)
    return nodes

def delete_elems(elems, deleted_nodes):
    """
    Deletes elements connected to deleted nodes
    """
    deleted_elements = []
    for z in deleted_nodes:
        for i in range(np.size(elems,axis=0)):
            if elems[i,0] == z or elems[i,1] == z:
                deleted_elements.append(i)
    elems_present = np.delete(elems, deleted_elements, axis=0)
    return elems_present

def get_edge_nodes(n_x, n_y, mode='dictionary'):
    """
    Returns nodes on the boundaries 
    """
    
    n_nodes = 2*n_x+(4*n_x+2)*n_y
    nodes_list = np.arange(n_nodes)
    
    nodes_bottom_edge = []
    nodes_left_edge = []
    nodes_right_edge = []
    nodes_top_edge = []
    
    for j in range(n_x):
        nodes_bottom_edge.append(j)
    
    for j in range(2*n_y):
        nodes_left_edge.append(n_x+(j)*(2*n_x+1))
    
    for j in range(2*n_y):
        nodes_right_edge.append(3*n_x+(j)*(2*n_x+1))
    
    nodes_top_edge.append(list(np.ravel(nodes_list[-n_x:])))
    
    nodes_top_edge = list(nodes_top_edge)[0]
    
    nodes_bottom_edge = np.array(np.sort(list(set(np.array(nodes_bottom_edge)))),dtype = np.int32)
    nodes_top_edge = np.array(np.sort(list(set(np.array(nodes_top_edge)))),dtype = np.int32)
    nodes_left_edge = np.array(np.sort(list(set(np.array(nodes_left_edge)))),dtype = np.int32)
    nodes_right_edge = np.array(np.sort(list(set(np.array(nodes_right_edge)))),dtype = np.int32)
    
    if mode=='tuple':  
        return nodes_bottom_edge, nodes_top_edge, nodes_left_edge, nodes_right_edge
    elif mode=='dictionary':
        ks = ['nodes_bottom_edge', 'nodes_top_edge', 'nodes_left_edge', 'nodes_right_edge']
        vs = [nodes_bottom_edge, nodes_top_edge, nodes_left_edge, nodes_right_edge]
        get_edge_nodes_dict = dict(zip(ks, vs))
        return get_edge_nodes_dict
    else:
        return nodes_bottom_edge, nodes_top_edge, nodes_left_edge, nodes_right_edge
        

def get_nodes_per_row(n_x, n_y, mode='dictionary'):
    
    n_nodes = 2*n_x+(4*n_x+2)*n_y
    nodes_list = np.arange(n_nodes)
    
    nodes_row_0 = np.array([[np.arange(n_x)]])
    nodes_row_1 = np.array([[np.arange(n_x, 3*n_x+2, 2)]])
    nodes_row_2 = np.array([[np.arange(n_x+1, 3*n_x, 2)]])
    nodes_row_3 = np.array([[nodes_row_2+2*n_x+1]])
    nodes_row_4 = np.array([[nodes_row_1+2*n_x+1]])
    
    for j in range(n_y-1):
            nodes_row_1 = np.concatenate((nodes_row_1, np.array([nodes_row_1[-1]+(4*n_x+2)])))
            nodes_row_2 = np.concatenate((nodes_row_2, np.array([nodes_row_2[-1]+(4*n_x+2)])))
            nodes_row_3 = np.concatenate((nodes_row_3, np.array([nodes_row_3[-1]+(4*n_x+2)])))
            nodes_row_4 = np.concatenate((nodes_row_4, np.array([nodes_row_4[-1]+(4*n_x+2)])))
    
    nodes_row_last = np.array([nodes_list[-n_x:]]).ravel()
    
    (nodes_row_0, nodes_row_1, nodes_row_2, \
        nodes_row_3, nodes_row_4, nodes_row_last)=\
            (nodes_row_0.ravel(), nodes_row_1.ravel(), nodes_row_2.ravel(), \
                nodes_row_3.ravel(), nodes_row_4.ravel(), nodes_row_last.ravel())
    
    if mode=='tuple':  
        return nodes_row_0, nodes_row_1, nodes_row_2, nodes_row_3, nodes_row_4, nodes_row_last
    elif mode=='dictionary':
        ks = ['nodes_row_0', 'nodes_row_1', 'nodes_row_2', 'nodes_row_3', 'nodes_row_4', 'nodes_row_last']
        vs = [nodes_row_0, nodes_row_1, nodes_row_2, nodes_row_3, nodes_row_4, nodes_row_last]
        get_nodes_per_row_dict = dict(zip(ks, vs))
        return get_nodes_per_row_dict
    else:
        return nodes_row_0, nodes_row_1, nodes_row_2, nodes_row_3, nodes_row_4, nodes_row_last



def get_dofs_per_row(n_x, n_y, mode='dictionary'):
    nodes_row_0, nodes_row_1, nodes_row_2,\
        nodes_row_3, nodes_row_4, nodes_row_last = get_nodes_per_row(n_x, n_y, mode='tuple')

    nodes_row_0_u, nodes_row_0_v, nodes_row_0_phi = nodes_row_0*3, nodes_row_0*3+1, nodes_row_0*3+2
    nodes_row_1_u, nodes_row_1_v, nodes_row_1_phi = nodes_row_1*3, nodes_row_1*3+1, nodes_row_1*3+2
    nodes_row_2_u, nodes_row_2_v, nodes_row_2_phi = nodes_row_2*3, nodes_row_2*3+1, nodes_row_2*3+2
    nodes_row_3_u, nodes_row_3_v, nodes_row_3_phi = nodes_row_3*3, nodes_row_3*3+1, nodes_row_3*3+2
    nodes_row_4_u, nodes_row_4_v, nodes_row_4_phi = nodes_row_4*3, nodes_row_4*3+1, nodes_row_4*3+2
    nodes_row_last_u, nodes_row_last_v, nodes_row_last_phi = nodes_row_last*3, nodes_row_last*3+1, nodes_row_last*3+2
    
    if mode=='tuple':  
        return nodes_row_0_u, nodes_row_0_v, nodes_row_0_phi,\
                    nodes_row_1_u, nodes_row_1_v, nodes_row_1_phi,\
                        nodes_row_2_u, nodes_row_2_v, nodes_row_2_phi,\
                            nodes_row_3_u, nodes_row_3_v, nodes_row_3_phi,\
                                nodes_row_4_u, nodes_row_4_v, nodes_row_4_phi,\
                                    nodes_row_last_u, nodes_row_last_v, nodes_row_last_phi   
    elif mode=='dictionary':
        ks = ['nodes_row_0_u', 'nodes_row_0_v', 'nodes_row_0_phi',\
                'nodes_row_1_u', 'nodes_row_1_v', 'nodes_row_1_phi',\
                    'nodes_row_2_u', 'nodes_row_2_v', 'nodes_row_2_phi',\
                        'nodes_row_3_u', 'nodes_row_3_v', 'nodes_row_3_phi',\
                            'nodes_row_4_u', 'nodes_row_4_v', 'nodes_row_4_phi',\
                                'nodes_row_last_u', 'nodes_row_last_v', 'nodes_row_last_phi']
    
        vs = [nodes_row_0_u, nodes_row_0_v, nodes_row_0_phi,\
                    nodes_row_1_u, nodes_row_1_v, nodes_row_1_phi,\
                        nodes_row_2_u, nodes_row_2_v, nodes_row_2_phi,\
                            nodes_row_3_u, nodes_row_3_v, nodes_row_3_phi,\
                                nodes_row_4_u, nodes_row_4_v, nodes_row_4_phi,\
                                    nodes_row_last_u, nodes_row_last_v, nodes_row_last_phi]
        get_dofs_per_row_dict = dict(zip(ks, vs))
        
        return get_dofs_per_row_dict
    
    else:  
        return nodes_row_0_u, nodes_row_0_v, nodes_row_0_phi,\
                    nodes_row_1_u, nodes_row_1_v, nodes_row_1_phi,\
                        nodes_row_2_u, nodes_row_2_v, nodes_row_2_phi,\
                            nodes_row_3_u, nodes_row_3_v, nodes_row_3_phi,\
                                nodes_row_4_u, nodes_row_4_v, nodes_row_4_phi,\
                                    nodes_row_last_u, nodes_row_last_v, nodes_row_last_phi 
    

# ddic = get_dofs_per_row_dictionary(1,1)
# print(ddic)
# nodes_row_0_u = ddic['nodes_row_0_u']
# print(nodes_row_0_u)


def get_nodes_per_column(n_x, n_y):
    
    # n_nodes = 2*n_x+(4*n_x+2)*n_y
    # nodes_list = np.arange(n_nodes)
    # nodes_col_0 = np.array([[np.arange(n_x)]]).ravel()
    
    pass
    

def get_edge_dofs(n_x, n_y, mode='dictionary'):
    nodes_bottom_edge, nodes_top_edge, nodes_left_edge, nodes_right_edge = get_edge_nodes(n_x, n_y, mode='tuple')
    
    u_bottom_edge = np.array(np.sort(list(set(np.array(nodes_bottom_edge)*3))),dtype = np.int32)
    v_bottom_edge = np.array(np.sort(list(set(np.array(nodes_bottom_edge)*3+1))),dtype = np.int32)
    phi_bottom_edge = np.array(np.sort(list(set(np.array(nodes_bottom_edge)*3+2))),dtype = np.int32)

    u_top_edge = np.array(np.sort(list(set(np.array(nodes_top_edge)*3))),dtype = np.int32)
    v_top_edge = np.array(np.sort(list(set(np.array(nodes_top_edge)*3+1))),dtype = np.int32)
    phi_top_edge = np.array(np.sort(list(set(np.array(nodes_top_edge)*3+2))),dtype = np.int32)

    u_left_edge = np.array(np.sort(list(set(np.array(nodes_left_edge)*3))),dtype = np.int32)
    v_left_edge = np.array(np.sort(list(set(np.array(nodes_left_edge)*3+1))),dtype = np.int32)
    phi_left_edge = np.array(np.sort(list(set(np.array(nodes_left_edge)*3+2))),dtype = np.int32)

    u_right_edge = np.array(np.sort(list(set(np.array(nodes_right_edge)*3))),dtype = np.int32)
    v_right_edge = np.array(np.sort(list(set(np.array(nodes_right_edge)*3+1))),dtype = np.int32)
    phi_right_edge = np.array(np.sort(list(set(np.array(nodes_right_edge)*3+2))),dtype = np.int32)
    
    if mode=='tuple':  
        return u_bottom_edge, v_bottom_edge, phi_bottom_edge,\
            u_top_edge, v_top_edge, phi_top_edge,\
                u_left_edge, v_left_edge, phi_left_edge,\
                    u_right_edge, v_right_edge, phi_right_edge
    
    elif mode=='dictionary':
        ks = ['u_bottom_edge', 'v_bottom_edge', 'phi_bottom_edge',\
            'u_top_edge', 'v_top_edge', 'phi_top_edge',\
                'u_left_edge', 'v_left_edge', 'phi_left_edge',\
                    'u_right_edge', 'v_right_edge', 'phi_right_edge']
    
        vs = [u_bottom_edge, v_bottom_edge, phi_bottom_edge,\
            u_top_edge, v_top_edge, phi_top_edge,\
                u_left_edge, v_left_edge, phi_left_edge,\
                    u_right_edge, v_right_edge, phi_right_edge]
        
        get_edge_dofs_dict = dict(zip(ks, vs))
        
        return get_edge_dofs_dict
    

def rotate_grid(XY, phi_1):
    rotmatr = np.array([[np.cos(phi_1),-np.sin(phi_1)],
                        [np.sin(phi_1),np.cos(phi_1)]])
    for j in range(np.size(XY, axis=0)):
        XY[j] = np.matmul(rotmatr, XY[j])
    return XY

def create_hoop(XY, R_0, phi_0):
    L_x = np.max(XY[:,0])
    L_y = np.max(XY[:,1])
    n_nodes = np.size(XY, axis=0)
    XY[:,0] += R_0
    R, phi = np.zeros(n_nodes), np.zeros(n_nodes)
    for j in range(n_nodes):
        R[j] = XY[j,0]
        phi[j] = phi_0*(XY[j,1]/L_y)
    for j in range(n_nodes):
        XY[j,0] = R[j]*np.cos(phi[j])
        XY[j,1] = R[j]*np.sin(phi[j])
    return XY
