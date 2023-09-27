import numpy as np

class D2_frame_node:
    def __init__(self, number, XY):
        self.number = number
        self.x = XY[0]
        self.y = XY[1]
        self.dofs = np.arange(self.number*3,self.number*3+3)
    def change_number(self, number_new):
        self.number = number_new
    def change_x(self, x_new):
        self.x = x_new
    def change_y(self, y_new):
        self.y = y_new

class D2_frame_element:
    def __init__(self, node_i, node_k, E, nu, rho, k, b, h):
        self.i_nn = node_i # i node number
        self.k_nn = node_k
        self.dofs = np.append(node_i.dofs, node_k.dofs)
        self.xi = node_i.x
        self.yi = node_i.y
        self.xk = node_k.x
        self.yk = node_k.y
        self.E = E
        self.nu = nu
        self.rho = rho
        self.k = k
        self.b = b
        self.h = h
        self.A = self.b * self.h
        self.J = 1/12 * self.b * self.h**3
        self.l = np.sqrt((self.xk-self.xi)**2+(self.yk-self.yi)**2)
    def change_E(self, E_new):
        self.E = E_new
    def change_nu(self, nu_new):
        self.nu = nu_new
    def change_rho(self, rho_new):
        self.rho = rho_new
    def change_k(self, k_new):
        self.k = k_new
    def change_b(self, b_new):
        self.b = b_new
        self.A = self.b * self.h
        self.J = 1/12 * self.b * self.h**3
    def change_h(self, h_new):
        self.h = h_new
        self.A = self.b * self.h
        self.J = 1/12 * self.b * self.h**3

def K_lin_loc(self):
    k = np.array([
    [self.E*self.A/self.l, 0, 0, -self.E*self.A/self.l, 0,0],
    [0, 12*self.E*self.J/self.l**3, 6*self.E*self.J/self.l**2, 0, -12*self.E*self.J/self.l**3, 6*self.E*self.J/self.l**2],
    [0,6*self.E*self.J/self.l**2, 4*self.E*self.J/self.l, 0, -6*self.E*self.J/self.l**2, 2*self.E*self.J/self.l],
    [-self.E*self.A/self.l, 0, 0, self.E*self.A/self.l, 0,0],
    [0, -12*self.E*self.J/self.l**3, -6*self.E*self.J/self.l**2, 0, 12*self.E*self.J/self.l**3, -6*self.E*self.J/self.l**2],
    [0,6*self.E*self.J/self.l**2, 2*self.E*self.J/self.l, 0, -6*self.E*self.J/self.l**2, 4*self.E*self.J/self.l]
    ])
    return k
D2_frame_element.K_lin_loc = K_lin_loc

def K_nlin_0_loc(self):
    k = 1/self.l*np.array([
    [0, 0, 0, 0, 0,0],
    [0, 6/5, self.l/10, 0, -6/5, self.l/10],
    [0, self.l/10, 2*self.l**2/15, 0, -self.l/10, -self.l**2/30],
    [0, 0, 0, 0, 0,0],
    [0, -6/5, -self.l/10, 0, 6/5, -self.l/10],
    [0, self.l/10, -self.l**2/30, 0, -self.l/10, 2*self.l**2/15],
    ])
    return k
D2_frame_element.K_nlin_0_loc = K_nlin_0_loc

def U_loc_glob(self, U_glob):
    return U_glob[self.dofs]
D2_frame_element.U_loc_glob = U_loc_glob

def U_loc_loc(self, U_glob):
    return self.Transf_Matr() @ U_glob[self.dofs]
D2_frame_element.U_loc_loc = U_loc_loc

def strain_energy(self, U_glob):
    energy = 1/2*(self.U_loc_loc(U_glob)).T @ (self.K_lin_loc() @ self.U_loc_loc(U_glob))
    return energy
D2_frame_element.strain_energy = strain_energy

def F_loc_loc(self, U_glob):
    return self.K_lin_loc() @ (self.Transf_Matr() @ U_glob[self.dofs])
D2_frame_element.F_loc_loc = F_loc_loc

def F_nlin_loc_glob(self, U_glob):
    q = self.Transf_Matr() @ U_glob[self.dofs]
    f_nlin = np.zeros(6)
    f_nlin[0] = self.A*self.E*(-2*self.l**2*q[2]**2 + self.l**2*q[2]*q[5] \
        - 2*self.l**2*q[5]**2 - 3*self.l*q[1]*q[2] - 3*self.l*q[1]*q[5] \
            + 3*self.l*q[2]*q[4] + 3*self.l*q[4]*q[5] - 18*q[1]**2 \
                + 36*q[1]*q[4] - 18*q[4]**2)/(30*self.l**2)
    f_nlin[1] = self.A*self.E*(-self.l**3*q[2]**3 + 3*self.l**3*q[2]**2*q[5] + 3*self.l**3*q[2]*q[5]**2 \
        - self.l**3*q[5]**3 - 28*self.l**2*q[0]*q[2] - 28*self.l**2*q[0]*q[5] + 36*self.l**2*q[1]*q[2]**2 \
            + 36*self.l**2*q[1]*q[5]**2 - 36*self.l**2*q[2]**2*q[4] + 28*self.l**2*q[2]*q[3] \
                + 28*self.l**2*q[3]*q[5] - 36*self.l**2*q[4]*q[5]**2 - 336*self.l*q[0]*q[1] \
                    + 336*self.l*q[0]*q[4] + 108*self.l*q[1]**2*q[2] + 108*self.l*q[1]**2*q[5] \
                        - 216*self.l*q[1]*q[2]*q[4] + 336*self.l*q[1]*q[3] - 216*self.l*q[1]*q[4]*q[5] \
                            + 108*self.l*q[2]*q[4]**2 - 336*self.l*q[3]*q[4] + 108*self.l*q[4]**2*q[5] \
                                + 288*q[1]**3 - 864*q[1]**2*q[4] + 864*q[1]*q[4]**2 \
                                    - 288*q[4]**3)/(280*self.l**3)
    f_nlin[2] = self.A*self.E*(24*self.l**3*q[2]**3 - 9*self.l**3*q[2]**2*q[5] + 6*self.l**3*q[2]*q[5]**2 - 3*self.l**3*q[5]**3 \
        - 112*self.l**2*q[0]*q[2] + 28*self.l**2*q[0]*q[5] - 9*self.l**2*q[1]*q[2]**2 + 18*self.l**2*q[1]*q[2]*q[5] \
            + 9*self.l**2*q[1]*q[5]**2 + 9*self.l**2*q[2]**2*q[4] + 112*self.l**2*q[2]*q[3] - 18*self.l**2*q[2]*q[4]*q[5] \
                - 28*self.l**2*q[3]*q[5] - 9*self.l**2*q[4]*q[5]**2 - 84*self.l*q[0]*q[1] + 84*self.l*q[0]*q[4] \
                    + 108*self.l*q[1]**2*q[2] - 216*self.l*q[1]*q[2]*q[4] + 84*self.l*q[1]*q[3] + 108*self.l*q[2]*q[4]**2 \
                        - 84*self.l*q[3]*q[4] + 108*q[1]**3 - 324*q[1]**2*q[4] + 324*q[1]*q[4]**2 \
                            - 108*q[4]**3)/(840*self.l**2)
    f_nlin[3] = self.A*self.E*(2*self.l**2*q[2]**2 - self.l**2*q[2]*q[5] + 2*self.l**2*q[5]**2 \
        + 3*self.l*q[1]*q[2] + 3*self.l*q[1]*q[5] - 3*self.l*q[2]*q[4] - 3*self.l*q[4]*q[5] \
            + 18*q[1]**2 - 36*q[1]*q[4] + 18*q[4]**2)/(30*self.l**2)
    f_nlin[4] = self.A*self.E*(self.l**3*q[2]**3 - 3*self.l**3*q[2]**2*q[5] - 3*self.l**3*q[2]*q[5]**2 \
        + self.l**3*q[5]**3 + 28*self.l**2*q[0]*q[2] + 28*self.l**2*q[0]*q[5] - 36*self.l**2*q[1]*q[2]**2 \
            - 36*self.l**2*q[1]*q[5]**2 + 36*self.l**2*q[2]**2*q[4] - 28*self.l**2*q[2]*q[3] \
                - 28*self.l**2*q[3]*q[5] + 36*self.l**2*q[4]*q[5]**2 + 336*self.l*q[0]*q[1] \
                    - 336*self.l*q[0]*q[4] - 108*self.l*q[1]**2*q[2] - 108*self.l*q[1]**2*q[5] \
                        + 216*self.l*q[1]*q[2]*q[4] - 336*self.l*q[1]*q[3] + 216*self.l*q[1]*q[4]*q[5] \
                            - 108*self.l*q[2]*q[4]**2 + 336*self.l*q[3]*q[4] - 108*self.l*q[4]**2*q[5] \
                                - 288*q[1]**3 + 864*q[1]**2*q[4] - 864*q[1]*q[4]**2 + 288*q[4]**3)/(280*self.l**3)
    f_nlin[5] = self.A*self.E*(-3*self.l**3*q[2]**3 + 6*self.l**3*q[2]**2*q[5] - 9*self.l**3*q[2]*q[5]**2 + 24*self.l**3*q[5]**3 + 28*self.l**2*q[0]*q[2] \
        - 112*self.l**2*q[0]*q[5] + 9*self.l**2*q[1]*q[2]**2 + 18*self.l**2*q[1]*q[2]*q[5] - 9*self.l**2*q[1]*q[5]**2 - 9*self.l**2*q[2]**2*q[4] \
            - 28*self.l**2*q[2]*q[3] - 18*self.l**2*q[2]*q[4]*q[5] + 112*self.l**2*q[3]*q[5] + 9*self.l**2*q[4]*q[5]**2 - 84*self.l*q[0]*q[1] \
                + 84*self.l*q[0]*q[4] + 108*self.l*q[1]**2*q[5] + 84*self.l*q[1]*q[3] - 216*self.l*q[1]*q[4]*q[5] \
                    - 84*self.l*q[3]*q[4] + 108*self.l*q[4]**2*q[5] + 108*q[1]**3 - 324*q[1]**2*q[4] + 324*q[1]*q[4]**2 \
                        - 108*q[4]**3)/(840*self.l**2)    
    return self.Transf_Matr().T @ f_nlin
D2_frame_element.F_nlin_loc_glob = F_nlin_loc_glob
    
def M_lin_loc(self):
    m = self.rho*self.l*np.array([
    [self.A/3,0,0,
        self.A/6,0,0],
    [0,13/35*self.A+6*self.J/(5*self.l**2),11/210*self.l*self.A+self.J/(10*self.l),
        0,9/70*self.A-6*self.J/(5*self.l**2),-13/420*self.l*self.A+self.J/(10*self.l)],
    [0,11/210*self.l*self.A+self.J/(10*self.l),1/105*self.A*self.l**2+2*self.J/(15),
        0,13/420*self.l*self.A-self.J/(10*self.l),-1/140*self.A*self.l**2-self.J/(30)],
    [self.A/6,0,0,
        self.A/3,0,0],
    [0,9/70*self.A-6*self.J/(5*self.l**2),13/420*self.l*self.A-self.J/(10*self.l),
        0,13/35*self.A+6*self.J/(5*self.l**2),-11/210*self.l*self.A-self.J/(10*self.l)],
    [0,-13/420*self.l*self.A+self.J/(10*self.l),-1/140*self.A*self.l**2-self.J/(30),
        0,-11/210*self.l*self.A-self.J/(10*self.l),1/105*self.A*self.l**2+2*self.J/(15)]
    ])
    return m
D2_frame_element.M_lin_loc = M_lin_loc

def Loc_to_Glob_Matr(self, n_dofs):
    # initialize
    T = np.identity(n_dofs)
    TT = T[:, self.dofs]
    return TT
D2_frame_element.Loc_to_Glob_Matr = Loc_to_Glob_Matr

def Transf_Matr(self):
    c = (self.xk-self.xi)/self.l
    s = (self.yk-self.yi)/self.l
    C_e = np.array([
        [c, s, 0, 0, 0, 0],
        [-s, c, 0,0,0,0,],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0],
        [0, 0, 0, -s, c,0],
        [0, 0, 0, 0, 0, 1]
    ])
    return C_e
D2_frame_element.Transf_Matr = Transf_Matr      

def K_lin_loc_glob(self):
    K_loc_glob = np.matmul(np.transpose(self.Transf_Matr()), np.matmul(self.K_lin_loc(), self.Transf_Matr()))
    return K_loc_glob
D2_frame_element.K_lin_loc_glob = K_lin_loc_glob

def M_lin_loc_glob(self):
    M_loc_glob = np.matmul(np.transpose(self.Transf_Matr()), np.matmul(self.M_lin_loc(), self.Transf_Matr()))
    return M_loc_glob
D2_frame_element.M_lin_loc_glob = M_lin_loc_glob


