import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Polygon
import sympy as sp
from matplotlib.collections import PatchCollection
from matplotlib import cm
from pickletools import float8
from scipy.interpolate import griddata
import scipy as scp
import scipy.linalg
import scipy.ndimage
import os

import scipy.linalg
from scipy.linalg import fractional_matrix_power

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Disc_node:
    '''
    Simply disc node, self-explanatory
    '''
    def __init__(self, number, XY):
        self.number = number
        self.x = XY[0]
        self.y = XY[1]
        self.dofs = np.arange(self.number*2,(self.number+1)*2)

class Disc_Element:
    """
    give properties in this order:
    E, nu, rho, h
    """
    def __init__(self,nodes_list, XY, E_, nu_, rho_, h_):
        self.nodes_list = nodes_list # ordered tuple
        self.coords = XY[self.nodes_list]
        self.x0, self.y0 = self.coords[0,0], self.coords[0,1]
        self.dofs_list = np.hstack([np.arange(2*j,2*(j+1)) for j in self.nodes_list])
        self.lx = np.max(self.coords[:,0])-np.min(self.coords[:,0])
        self.ly = np.max(self.coords[:,1])-np.min(self.coords[:,1])
        self.E = E_
        self.nu = nu_
        self.rho = rho_
        self.h = h_
        self.mu = self.rho*self.h
    def change_E(self, E_new):
        self.E = E_new
    def change_nu(self, nu_new):
        self.nu = nu_new
    def change_rho(self, rho_new):
        self.rho = rho_new
        self.mu = self.rho*self.h
    def change_h(self, h_new):
        self.h = h_new
        self.mu = self.rho*self.h

# class D2_pl_str_elem():
#     def __init__(self, list_x, list_y, E, h, nu, rho):
#         self.list_x = list_x
#         self.list_y = list_y
#         self.E = E
#         self.h = h
#         self.nu = nu
#         self.rho = rho
#         self.a = 1/2*(np.max(self.list_x)-np.min(self.list_x))
#         self.b = 1/2*(np.max(self.list_y)-np.min(self.list_y))
#         self.x_C = np.min(self.list_x) + self.a
#         self.y_C = np.min(self.list_y) + self.b
#     def change_E(self, E_new):
#         self.E = E_new
#     def change_nu(self, nu_new):
#         self.nu = nu_new
#     def change_rho(self, rho_new):
#         self.rho = rho_new
#     def change_h(self, h_new):
#         self.h = h_new
#         self.A = self.b * self.h
#         self.J = 1/12 * self.b * self.h**3

def K_loc(self):
        k = np.array([[-2*self.lx*self.nu/self.ly + 2*self.lx/self.ly + 4*self.ly/self.lx, 3*self.nu/2 + 3/2, -self.lx*self.nu/self.ly + self.lx/self.ly - 4*self.ly/self.lx, 9*self.nu/2 - 3/2, self.lx*self.nu/self.ly - self.lx/self.ly - 2*self.ly/self.lx, -3*self.nu/2 - 3/2, 2*self.lx*self.nu/self.ly - 2*self.lx/self.ly + 2*self.ly/self.lx, 3/2 - 9*self.nu/2], 
                      [3*self.nu/2 + 3/2, 4*self.lx/self.ly - 2*self.ly*self.nu/self.lx + 2*self.ly/self.lx, 3/2 - 9*self.nu/2, 2*self.lx/self.ly + 2*self.ly*self.nu/self.lx - 2*self.ly/self.lx, -3*self.nu/2 - 3/2, -2*self.lx/self.ly + self.ly*self.nu/self.lx - self.ly/self.lx, 9*self.nu/2 - 3/2, -4*self.lx/self.ly - self.ly*self.nu/self.lx + self.ly/self.lx], 
                      [-self.lx*self.nu/self.ly + self.lx/self.ly - 4*self.ly/self.lx, 3/2 - 9*self.nu/2, -2*self.lx*self.nu/self.ly + 2*self.lx/self.ly + 4*self.ly/self.lx, -3*self.nu/2 - 3/2, 2*self.lx*self.nu/self.ly - 2*self.lx/self.ly + 2*self.ly/self.lx, 9*self.nu/2 - 3/2, self.lx*self.nu/self.ly - self.lx/self.ly - 2*self.ly/self.lx, 3*self.nu/2 + 3/2], 
                      [9*self.nu/2 - 3/2, 2*self.lx/self.ly + 2*self.ly*self.nu/self.lx - 2*self.ly/self.lx, -3*self.nu/2 - 3/2, 4*self.lx/self.ly - 2*self.ly*self.nu/self.lx + 2*self.ly/self.lx, 3/2 - 9*self.nu/2, -4*self.lx/self.ly - self.ly*self.nu/self.lx + self.ly/self.lx, 3*self.nu/2 + 3/2, -2*self.lx/self.ly + self.ly*self.nu/self.lx - self.ly/self.lx], 
                      [self.lx*self.nu/self.ly - self.lx/self.ly - 2*self.ly/self.lx, -3*self.nu/2 - 3/2, 2*self.lx*self.nu/self.ly - 2*self.lx/self.ly + 2*self.ly/self.lx, 3/2 - 9*self.nu/2, -2*self.lx*self.nu/self.ly + 2*self.lx/self.ly + 4*self.ly/self.lx, 3*self.nu/2 + 3/2, -self.lx*self.nu/self.ly + self.lx/self.ly - 4*self.ly/self.lx, 9*self.nu/2 - 3/2], 
                      [-3*self.nu/2 - 3/2, -2*self.lx/self.ly + self.ly*self.nu/self.lx - self.ly/self.lx, 9*self.nu/2 - 3/2, -4*self.lx/self.ly - self.ly*self.nu/self.lx + self.ly/self.lx, 3*self.nu/2 + 3/2, 4*self.lx/self.ly - 2*self.ly*self.nu/self.lx + 2*self.ly/self.lx, 3/2 - 9*self.nu/2, 2*self.lx/self.ly + 2*self.ly*self.nu/self.lx - 2*self.ly/self.lx], 
                      [2*self.lx*self.nu/self.ly - 2*self.lx/self.ly + 2*self.ly/self.lx, 9*self.nu/2 - 3/2, self.lx*self.nu/self.ly - self.lx/self.ly - 2*self.ly/self.lx, 3*self.nu/2 + 3/2, -self.lx*self.nu/self.ly + self.lx/self.ly - 4*self.ly/self.lx, 3/2 - 9*self.nu/2, -2*self.lx*self.nu/self.ly + 2*self.lx/self.ly + 4*self.ly/self.lx, -3*self.nu/2 - 3/2], 
                      [3/2 - 9*self.nu/2, -4*self.lx/self.ly - self.ly*self.nu/self.lx + self.ly/self.lx, 3*self.nu/2 + 3/2, -2*self.lx/self.ly + self.ly*self.nu/self.lx - self.ly/self.lx, 9*self.nu/2 - 3/2, 2*self.lx/self.ly + 2*self.ly*self.nu/self.lx - 2*self.ly/self.lx, -3*self.nu/2 - 3/2, 4*self.lx/self.ly - 2*self.ly*self.nu/self.lx + 2*self.ly/self.lx]
                      ])
        kk = k*self.E*self.h/(12*(1-self.nu**2))
        return kk
Disc_Element.K_loc = K_loc

def M_loc(self):
        
        m = self.lx*self.ly*self.h*self.rho*np.array(
            [[1/9, 0, 1/18, 0, 1/36, 0, 1/18, 0], 
             [0, 1/9, 0, 1/18, 0, 1/36, 0, 1/18], 
             [1/18, 0, 1/9, 0, 1/18, 0, 1/36, 0], 
             [0, 1/18, 0, 1/9, 0, 1/18, 0, 1/36], 
             [1/36, 0, 1/18, 0, 1/9, 0, 1/18, 0], 
             [0, 1/36, 0, 1/18, 0, 1/9, 0, 1/18], 
             [1/18, 0, 1/36, 0, 1/18, 0, 1/9, 0], 
             [0, 1/18, 0, 1/36, 0, 1/18, 0, 1/9]])

        return m
Disc_Element.M_loc = M_loc

def U_loc(self, U_glob):
    return U_glob[self.dofs_list]
Disc_Element.U_loc = U_loc

def Node_Stress(self, U_glob):
    
    u_0, v_0, u_1, v_1, u_2, v_2, u_3, v_3 = self.U_loc(U_glob)
    
    s_x_00 = -self.E*(self.nu*(-v_0/self.ly + v_3/self.ly) - u_0/self.lx + u_1/self.lx)/(self.nu**2 - 1)
    s_x_01 = -self.E*(self.nu*(-v_1/self.ly + v_2/self.ly) - u_0/self.lx + u_1/self.lx)/(self.nu**2 - 1)
    s_x_10 = -self.E*(self.nu*(-v_1/self.ly + v_2/self.ly) + u_2/self.lx - u_3/self.lx)/(self.nu**2 - 1)
    s_x_11 = -self.E*(self.nu*(-v_0/self.ly + v_3/self.ly) + u_2/self.lx - u_3/self.lx)/(self.nu**2 - 1)
    
    s_y_00 = -self.E*(self.nu*(-u_0/self.lx + u_1/self.lx) - v_0/self.ly + v_3/self.ly)/(self.nu**2 - 1)
    s_y_01 = -self.E*(self.nu*(-u_0/self.lx + u_1/self.lx) - v_1/self.ly + v_2/self.ly)/(self.nu**2 - 1)
    s_y_10 = -self.E*(self.nu*(u_2/self.lx - u_3/self.lx) - v_1/self.ly + v_2/self.ly)/(self.nu**2 - 1)
    s_y_11 = -self.E*(self.nu*(u_2/self.lx - u_3/self.lx) - v_0/self.ly + v_3/self.ly)/(self.nu**2 - 1)

    s_xy_00 = 2*self.E*(-u_0/(2*self.ly) + u_3/(2*self.ly) - v_0/(2*self.lx) + v_1/(2*self.lx))/(2*self.nu + 2)
    s_xy_01 = 2*self.E*(-u_1/(2*self.ly) + u_2/(2*self.ly) - v_0/(2*self.lx) + v_1/(2*self.lx))/(2*self.nu + 2)
    s_xy_10 = 2*self.E*(-u_1/(2*self.ly) + u_2/(2*self.ly) + v_2/(2*self.lx) - v_3/(2*self.lx))/(2*self.nu + 2)
    s_xy_11 = 2*self.E*(-u_0/(2*self.ly) + u_3/(2*self.ly) + v_2/(2*self.lx) - v_3/(2*self.lx))/(2*self.nu + 2)

    return np.array([s_x_00, s_x_01, s_x_10, s_x_11]),\
                np.array([s_y_00, s_y_01, s_y_10, s_y_11]),\
                    np.array([s_xy_00, s_xy_01, s_xy_10, s_xy_11])

Disc_Element.Node_Stress = Node_Stress










# def deflections(self, u, v, numnum):
#     ae, be = self.a, self.b
#     xCe, yCe = self.x_C, self.y_C
#     xe, ye = np.linspace(xCe-ae, xCe+ae, num = numnum), np.linspace(yCe-be, yCe+be, num = numnum)
#     xxe, yye = np.meshgrid(xe, ye)
#     N_0 = 0.25*(1+(xxe-xCe)/ae)*(1+(yye-yCe)/be)
#     N_1 = 0.25*(1+(xxe-xCe)/ae)*(1-(yye-yCe)/be)
#     N_2 = 0.25*(1-(xxe-xCe)/ae)*(1+(yye-yCe)/be)
#     N_3 = 0.25*(1-(xxe-xCe)/ae)*(1-(yye-yCe)/be)
#     NN = np.stack((N_0, N_1, N_2, N_3))
#     Ue = np.einsum('ijk,i', NN, u)
#     Ve = np.einsum('ijk,i', NN, v)
#     sols_e = np.stack((xxe, yye, Ue, Ve))
#     return sols_e

# def stress(self, u, v, numnum):
#     ae, be = self.a, self.b
#     xCe, yCe = self.x_C, self.y_C
#     xe, ye = np.linspace(xCe-ae, xCe+ae, num = numnum), np.linspace(yCe-be, yCe+be, num = numnum)
#     xxe, yye = np.meshgrid(xe, ye)
#     dN_0dx = 0.25*(1/ae)*(1+(yye-yCe)/be)
#     dN_1dx = 0.25*(1/ae)*(1-(yye-yCe)/be)
#     dN_2dx = -0.25*(1/ae)*(1+(yye-yCe)/be)
#     dN_3dx = -0.25*(1/ae)*(1-(yye-yCe)/be)
#     dN_0dy = 0.25*(1+(xxe-xCe)/ae)*(1/be)
#     dN_1dy = -0.25*(1+(xxe-xCe)/ae)*(1/be)
#     dN_2dy = 0.25*(1-(xxe-xCe)/ae)*(1/be)
#     dN_3dy = -0.25*(1-(xxe-xCe)/ae)*(1/be)
#     dNdx = np.stack((dN_0dx, dN_1dx, dN_2dx, dN_3dx))
#     dNdy = np.stack((dN_0dy, dN_1dy, dN_2dy, dN_3dy))
#     sigma_x = -self.E*(np.einsum('ijk,i', dNdx, u)+self.nu*np.einsum('ijk,i', dNdy, v))/(self.nu**2-1)
#     sigma_y = -self.E*(self.nu*np.einsum('ijk,i', dNdx, u)+np.einsum('ijk,i', dNdy, v))/(self.nu**2-1)
#     tau_xy = 2*self.E/(2+2*self.nu)*(np.einsum('ijk,i', dNdy, u)+np.einsum('ijk,i', dNdx, v))*1/2
#     sigma_1 = 1/2*(sigma_x+sigma_y)+1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sigma_2 = 1/2*(sigma_x+sigma_y)-1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sols_e = np.stack((xxe, yye, sigma_x, sigma_y, tau_xy, sigma_1, sigma_2))
#     return sols_e

# def stress_avg(self, u, v, numnum):
#     ae, be = self.a, self.b
#     xCe, yCe = self.x_C, self.y_C
#     xe, ye = np.linspace(xCe-ae, xCe+ae, num = numnum), np.linspace(yCe-be, yCe+be, num = numnum)
#     xxe, yye = np.meshgrid(xe, ye)
#     dN_0dx = xxe*0+0.25*(1/ae)
#     dN_1dx = xxe*0+0.25*(1/ae)
#     dN_2dx = xxe*0-0.25*(1/ae)
#     dN_3dx = xxe*0-0.25*(1/ae)
#     dN_0dy = xxe*0+0.25*(1/be)
#     dN_1dy = xxe*0-0.25*(1/be)
#     dN_2dy = xxe*0+0.25*(1/be)
#     dN_3dy = xxe*0-0.25*(1/be)
#     dNdx = np.stack((dN_0dx, dN_1dx, dN_2dx, dN_3dx))
#     dNdy = np.stack((dN_0dy, dN_1dy, dN_2dy, dN_3dy))
#     sigma_x = -self.E*(np.einsum('ijk,i', dNdx, u)+self.nu*np.einsum('ijk,i', dNdy, v))/(self.nu**2-1)
#     sigma_y = -self.E*(self.nu*np.einsum('ijk,i', dNdx, u)+np.einsum('ijk,i', dNdy, v))/(self.nu**2-1)
#     tau_xy = 2*self.E/(2+2*self.nu)*(np.einsum('ijk,i', dNdy, u)+np.einsum('ijk,i', dNdx, v))*1/2
#     sigma_1 = 1/2*(sigma_x+sigma_y)+1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sigma_2 = 1/2*(sigma_x+sigma_y)-1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sols_e = np.stack((xxe, yye, sigma_x, sigma_y, tau_xy, sigma_1, sigma_2))
#     return sols_e

# def stress_smooth(self, u, v):
#     ae, be = self.a, self.b
#     xCe, yCe = self.x_C, self.y_C
#     dN_0dx = 0.25*(1/ae)
#     dN_1dx = 0.25*(1/ae)
#     dN_2dx = -0.25*(1/ae)
#     dN_3dx = -0.25*(1/ae)
#     dN_0dy = 0.25*(1/be)
#     dN_1dy = -0.25*(1/be)
#     dN_2dy = 0.25*(1/be)
#     dN_3dy = -0.25*(1/be)
#     dNdx = np.stack((dN_0dx, dN_1dx, dN_2dx, dN_3dx))
#     dNdy = np.stack((dN_0dy, dN_1dy, dN_2dy, dN_3dy))
#     sigma_x = -self.E*(np.einsum('i,i', dNdx, u)+self.nu*np.einsum('i,i', dNdy, v))/(self.nu**2-1)
#     sigma_y = -self.E*(self.nu*np.einsum('i,i', dNdx, u)+np.einsum('i,i', dNdy, v))/(self.nu**2-1)
#     tau_xy = 2*self.E/(2+2*self.nu)*(np.einsum('i,i', dNdy, u)+np.einsum('i,i', dNdx, v))*1/2
#     sigma_1 = 1/2*(sigma_x+sigma_y)+1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sigma_2 = 1/2*(sigma_x+sigma_y)-1/2*np.sqrt((sigma_x-sigma_y)**2+4*tau_xy**2)
#     sols_e = xCe, yCe, sigma_x, sigma_y, tau_xy, sigma_1, sigma_2
#     return sols_e

# def constant_fc_over_elem(self, f_):
#     ae, be = self.a, self.b
#     xCe, yCe = self.x_C, self.y_C
#     xe, ye = np.linspace(xCe-ae, xCe+ae, num = 2), np.linspace(yCe-be, yCe+be, num = 2)
#     xxe, yye = np.meshgrid(xe, ye)
#     ge = (xxe*0 + yye*0)+f_
#     sols_e = np.stack((xxe, yye, ge))
#     return sols_e

# D2_pl_str_elem.deflections = deflections
# D2_pl_str_elem.stress = stress
# D2_pl_str_elem.stress_avg = stress_avg
# D2_pl_str_elem.stress_smooth = stress_smooth
# D2_pl_str_elem.constant_fc_over_elem = constant_fc_over_elem

