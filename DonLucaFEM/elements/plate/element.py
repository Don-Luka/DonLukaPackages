
# PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import eigh



import scipy.linalg
import sys
import os

import random
from sympy.matrices.expressions import inverse

import time
random.seed(time.time())

# import numba

# CLASSES

class Plate_node:
    '''
    Simply plate node, self-explanatory
    '''
    def __init__(self, number, XY):
        self.number = number
        self.x = XY[0]
        self.y = XY[1]
        self.dofs = np.arange(self.number*3,self.number*3+3)

class Thin_Plate_Element:
    """
    give properties in this order:
    E, nu, rho, h
    """
    def __init__(self,nodes_list, XY, E_, nu_, rho_, h_):
        self.nodes_list = nodes_list # ordered tuple
        self.coords = XY[self.nodes_list]
        self.x0, self.y0 = self.coords[0,0], self.coords[0,1]
        self.dofs_list = np.hstack([np.arange(3*j,3*(j+1)) for j in self.nodes_list])
        self.lx = np.max(self.coords[:,0])-np.min(self.coords[:,0])
        self.ly = np.max(self.coords[:,1])-np.min(self.coords[:,1])
        self.E = E_
        self.nu = nu_
        self.rho = rho_
        self.h = h_
        self.D = (self.E*self.h**3)/(12*(1-self.nu**2))
        self.mu = self.rho*self.h
    def change_E(self, E_new):
        self.E = E_new
        self.D = (self.E*self.h**3)/(12*(1-self.nu**2))
    def change_nu(self, nu_new):
        self.nu = nu_new
    def change_rho(self, rho_new):
        self.rho = rho_new
        self.mu = self.rho*self.h
    def change_h(self, h_new):
        self.h = h_new
        self.D = (self.E*self.h**3)/(12*(1-self.nu**2))
        self.mu = self.rho*self.h

def K_lin_loc_e(self):
    K = np.array(
        [
        [2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 + 10*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), -self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 - 10*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 + 5*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), -self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 - 5*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly)], [self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), 4*self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), -self.D*self.nu, self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), 2*self.D*(5*self.lx**2 + 2*self.ly**2*self.nu - 2*self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(10*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(15*self.lx*self.ly), 0], [-self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), -self.D*self.nu, -4*self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), -self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 0, self.D*(self.lx**2*self.nu - self.lx**2 + 10*self.ly**2)/(15*self.lx*self.ly), self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), 0, -self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), 0, 2*self.D*(2*self.lx**2*self.nu - 2*self.lx**2 + 5*self.ly**2)/(15*self.lx*self.ly)], [2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 - 10*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), -self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 + 10*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 - 5*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), -self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 + 5*self.ly**4)/(5*self.lx**3*self.ly**3), self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly)], [self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), 2*self.D*(5*self.lx**2 + 2*self.ly**2*self.nu - 2*self.ly**2)/(15*self.lx*self.ly), 0, self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), 4*self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), self.D*self.nu, -self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(10*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), 0], [self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 0, self.D*(self.lx**2*self.nu - self.lx**2 + 10*self.ly**2)/(15*self.lx*self.ly), self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), self.D*self.nu, -4*self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), -self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), 0, 2*self.D*(2*self.lx**2*self.nu - 2*self.lx**2 + 5*self.ly**2)/(15*self.lx*self.ly), -self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), 0, -self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly)], [-2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 + 5*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 - 5*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), -self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 + 10*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 - 10*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), -self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly)], [self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), 0, self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(10*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), 4*self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), -self.D*self.nu, -self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), 2*self.D*(5*self.lx**2 + 2*self.ly**2*self.nu - 2*self.ly**2)/(15*self.lx*self.ly), 0], [-self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), 0, -self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), -self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), 0, 2*self.D*(2*self.lx**2*self.nu - 2*self.lx**2 + 5*self.ly**2)/(15*self.lx*self.ly), self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), -self.D*self.nu, -4*self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 0, self.D*(self.lx**2*self.nu - self.lx**2 + 10*self.ly**2)/(15*self.lx*self.ly)], [-2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 - 5*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), -2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 + 5*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), -self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(5*self.lx**4 + 2*self.lx**2*self.ly**2*self.nu - 7*self.lx**2*self.ly**2 - 10*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 2*self.D*(10*self.lx**4 - 2*self.lx**2*self.ly**2*self.nu + 7*self.lx**2*self.ly**2 + 10*self.ly**4)/(5*self.lx**3*self.ly**3), -self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), -self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly)], [self.D*(10*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), self.D*(10*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(15*self.lx*self.ly), 0, self.D*(5*self.lx**2 + self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(5*self.lx**2 - 4*self.ly**2*self.nu - self.ly**2)/(5*self.lx*self.ly**2), 2*self.D*(5*self.lx**2 + 2*self.ly**2*self.nu - 2*self.ly**2)/(15*self.lx*self.ly), 0, -self.D*(10*self.lx**2 + 4*self.ly**2*self.nu + self.ly**2)/(5*self.lx*self.ly**2), 4*self.D*(5*self.lx**2 - self.ly**2*self.nu + self.ly**2)/(15*self.lx*self.ly), self.D*self.nu], [self.D*(4*self.lx**2*self.nu + self.lx**2 - 5*self.ly**2)/(5*self.lx**2*self.ly), 0, 2*self.D*(2*self.lx**2*self.nu - 2*self.lx**2 + 5*self.ly**2)/(15*self.lx*self.ly), self.D*(self.lx**2*self.nu - self.lx**2 + 5*self.ly**2)/(5*self.lx**2*self.ly), 0, -self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly), -self.D*(self.lx**2*self.nu - self.lx**2 - 10*self.ly**2)/(5*self.lx**2*self.ly), 0, self.D*(self.lx**2*self.nu - self.lx**2 + 10*self.ly**2)/(15*self.lx*self.ly), -self.D*(4*self.lx**2*self.nu + self.lx**2 + 10*self.ly**2)/(5*self.lx**2*self.ly), self.D*self.nu, -4*self.D*(self.lx**2*self.nu - self.lx**2 - 5*self.ly**2)/(15*self.lx*self.ly)]
        ]
        ,dtype=np.float64)
    return K
Thin_Plate_Element.K_lin_loc_e = K_lin_loc_e

def M_loc_e(self):
    M =np.array(
        [
        [1727*self.lx*self.ly*self.mu/12600, 461*self.lx*self.ly**2*self.mu/25200, -461*self.lx**2*self.ly*self.mu/25200, 613*self.lx*self.ly*self.mu/12600, 199*self.lx*self.ly**2*self.mu/25200, 137*self.lx**2*self.ly*self.mu/12600, 197*self.lx*self.ly*self.mu/12600, -29*self.lx*self.ly**2*self.mu/6300, 29*self.lx**2*self.ly*self.mu/6300, 613*self.lx*self.ly*self.mu/12600, -137*self.lx*self.ly**2*self.mu/12600, -199*self.lx**2*self.ly*self.mu/25200], [461*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/315, -self.lx**2*self.ly**2*self.mu/400, 199*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/630, self.lx**2*self.ly**2*self.mu/600, 29*self.lx*self.ly**2*self.mu/6300, -self.lx*self.ly**3*self.mu/840, self.lx**2*self.ly**2*self.mu/900, 137*self.lx*self.ly**2*self.mu/12600, -self.lx*self.ly**3*self.mu/420, -self.lx**2*self.ly**2*self.mu/600], [-461*self.lx**2*self.ly*self.mu/25200, -self.lx**2*self.ly**2*self.mu/400, self.lx**3*self.ly*self.mu/315, -137*self.lx**2*self.ly*self.mu/12600, -self.lx**2*self.ly**2*self.mu/600, -self.lx**3*self.ly*self.mu/420, -29*self.lx**2*self.ly*self.mu/6300, self.lx**2*self.ly**2*self.mu/900, -self.lx**3*self.ly*self.mu/840, -199*self.lx**2*self.ly*self.mu/25200, self.lx**2*self.ly**2*self.mu/600, self.lx**3*self.ly*self.mu/630], [613*self.lx*self.ly*self.mu/12600, 199*self.lx*self.ly**2*self.mu/25200, -137*self.lx**2*self.ly*self.mu/12600, 1727*self.lx*self.ly*self.mu/12600, 461*self.lx*self.ly**2*self.mu/25200, 461*self.lx**2*self.ly*self.mu/25200, 613*self.lx*self.ly*self.mu/12600, -137*self.lx*self.ly**2*self.mu/12600, 199*self.lx**2*self.ly*self.mu/25200, 197*self.lx*self.ly*self.mu/12600, -29*self.lx*self.ly**2*self.mu/6300, -29*self.lx**2*self.ly*self.mu/6300], [199*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/630, -self.lx**2*self.ly**2*self.mu/600, 461*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/315, self.lx**2*self.ly**2*self.mu/400, 137*self.lx*self.ly**2*self.mu/12600, -self.lx*self.ly**3*self.mu/420, self.lx**2*self.ly**2*self.mu/600, 29*self.lx*self.ly**2*self.mu/6300, -self.lx*self.ly**3*self.mu/840, -self.lx**2*self.ly**2*self.mu/900], [137*self.lx**2*self.ly*self.mu/12600, self.lx**2*self.ly**2*self.mu/600, -self.lx**3*self.ly*self.mu/420, 461*self.lx**2*self.ly*self.mu/25200, self.lx**2*self.ly**2*self.mu/400, self.lx**3*self.ly*self.mu/315, 199*self.lx**2*self.ly*self.mu/25200, -self.lx**2*self.ly**2*self.mu/600, self.lx**3*self.ly*self.mu/630, 29*self.lx**2*self.ly*self.mu/6300, -self.lx**2*self.ly**2*self.mu/900, -self.lx**3*self.ly*self.mu/840], [197*self.lx*self.ly*self.mu/12600, 29*self.lx*self.ly**2*self.mu/6300, -29*self.lx**2*self.ly*self.mu/6300, 613*self.lx*self.ly*self.mu/12600, 137*self.lx*self.ly**2*self.mu/12600, 199*self.lx**2*self.ly*self.mu/25200, 1727*self.lx*self.ly*self.mu/12600, -461*self.lx*self.ly**2*self.mu/25200, 461*self.lx**2*self.ly*self.mu/25200, 613*self.lx*self.ly*self.mu/12600, -199*self.lx*self.ly**2*self.mu/25200, -137*self.lx**2*self.ly*self.mu/12600], [-29*self.lx*self.ly**2*self.mu/6300, -self.lx*self.ly**3*self.mu/840, self.lx**2*self.ly**2*self.mu/900, -137*self.lx*self.ly**2*self.mu/12600, -self.lx*self.ly**3*self.mu/420, -self.lx**2*self.ly**2*self.mu/600, -461*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/315, -self.lx**2*self.ly**2*self.mu/400, -199*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/630, self.lx**2*self.ly**2*self.mu/600], [29*self.lx**2*self.ly*self.mu/6300, self.lx**2*self.ly**2*self.mu/900, -self.lx**3*self.ly*self.mu/840, 199*self.lx**2*self.ly*self.mu/25200, self.lx**2*self.ly**2*self.mu/600, self.lx**3*self.ly*self.mu/630, 461*self.lx**2*self.ly*self.mu/25200, -self.lx**2*self.ly**2*self.mu/400, self.lx**3*self.ly*self.mu/315, 137*self.lx**2*self.ly*self.mu/12600, -self.lx**2*self.ly**2*self.mu/600, -self.lx**3*self.ly*self.mu/420], [613*self.lx*self.ly*self.mu/12600, 137*self.lx*self.ly**2*self.mu/12600, -199*self.lx**2*self.ly*self.mu/25200, 197*self.lx*self.ly*self.mu/12600, 29*self.lx*self.ly**2*self.mu/6300, 29*self.lx**2*self.ly*self.mu/6300, 613*self.lx*self.ly*self.mu/12600, -199*self.lx*self.ly**2*self.mu/25200, 137*self.lx**2*self.ly*self.mu/12600, 1727*self.lx*self.ly*self.mu/12600, -461*self.lx*self.ly**2*self.mu/25200, -461*self.lx**2*self.ly*self.mu/25200], [-137*self.lx*self.ly**2*self.mu/12600, -self.lx*self.ly**3*self.mu/420, self.lx**2*self.ly**2*self.mu/600, -29*self.lx*self.ly**2*self.mu/6300, -self.lx*self.ly**3*self.mu/840, -self.lx**2*self.ly**2*self.mu/900, -199*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/630, -self.lx**2*self.ly**2*self.mu/600, -461*self.lx*self.ly**2*self.mu/25200, self.lx*self.ly**3*self.mu/315, self.lx**2*self.ly**2*self.mu/400], [-199*self.lx**2*self.ly*self.mu/25200, -self.lx**2*self.ly**2*self.mu/600, self.lx**3*self.ly*self.mu/630, -29*self.lx**2*self.ly*self.mu/6300, -self.lx**2*self.ly**2*self.mu/900, -self.lx**3*self.ly*self.mu/840, -137*self.lx**2*self.ly*self.mu/12600, self.lx**2*self.ly**2*self.mu/600, -self.lx**3*self.ly*self.mu/420, -461*self.lx**2*self.ly*self.mu/25200, self.lx**2*self.ly**2*self.mu/400, self.lx**3*self.ly*self.mu/315]
        ]
        ,dtype=np.float64)
    return M
Thin_Plate_Element.M_loc_e = M_loc_e

def K_nl_0_N_loc_e(self, N_xx, N_yy, N_xy):
    K_nl_0_Nxx = np.array([[46*self.ly/(105*self.lx), 11*self.ly**2/(210*self.lx), -self.ly/30, -46*self.ly/(105*self.lx), -11*self.ly**2/(210*self.lx), -self.ly/30, -17*self.ly/(105*self.lx), 13*self.ly**2/(420*self.lx), -self.ly/60, 17*self.ly/(105*self.lx), -13*self.ly**2/(420*self.lx), -self.ly/60], [11*self.ly**2/(210*self.lx), self.ly**3/(105*self.lx), 0, -11*self.ly**2/(210*self.lx), -self.ly**3/(105*self.lx), 0, -13*self.ly**2/(420*self.lx), self.ly**3/(140*self.lx), 0, 13*self.ly**2/(420*self.lx), -self.ly**3/(140*self.lx), 0], [-self.ly/30, 0, 2*self.lx*self.ly/45, self.ly/30, 0, -self.lx*self.ly/90, self.ly/60, 0, -self.lx*self.ly/180, -self.ly/60, 0, self.lx*self.ly/45], [-46*self.ly/(105*self.lx), -11*self.ly**2/(210*self.lx), self.ly/30, 46*self.ly/(105*self.lx), 11*self.ly**2/(210*self.lx), self.ly/30, 17*self.ly/(105*self.lx), -13*self.ly**2/(420*self.lx), self.ly/60, -17*self.ly/(105*self.lx), 13*self.ly**2/(420*self.lx), self.ly/60], [-11*self.ly**2/(210*self.lx), -self.ly**3/(105*self.lx), 0, 11*self.ly**2/(210*self.lx), self.ly**3/(105*self.lx), 0, 13*self.ly**2/(420*self.lx), -self.ly**3/(140*self.lx), 0, -13*self.ly**2/(420*self.lx), self.ly**3/(140*self.lx), 0], [-self.ly/30, 0, -self.lx*self.ly/90, self.ly/30, 0, 2*self.lx*self.ly/45, self.ly/60, 0, self.lx*self.ly/45, -self.ly/60, 0, -self.lx*self.ly/180], [-17*self.ly/(105*self.lx), -13*self.ly**2/(420*self.lx), self.ly/60, 17*self.ly/(105*self.lx), 13*self.ly**2/(420*self.lx), self.ly/60, 46*self.ly/(105*self.lx), -11*self.ly**2/(210*self.lx), self.ly/30, -46*self.ly/(105*self.lx), 11*self.ly**2/(210*self.lx), self.ly/30], [13*self.ly**2/(420*self.lx), self.ly**3/(140*self.lx), 0, -13*self.ly**2/(420*self.lx), -self.ly**3/(140*self.lx), 0, -11*self.ly**2/(210*self.lx), self.ly**3/(105*self.lx), 0, 11*self.ly**2/(210*self.lx), -self.ly**3/(105*self.lx), 0], [-self.ly/60, 0, -self.lx*self.ly/180, self.ly/60, 0, self.lx*self.ly/45, self.ly/30, 0, 2*self.lx*self.ly/45, -self.ly/30, 0, -self.lx*self.ly/90], [17*self.ly/(105*self.lx), 13*self.ly**2/(420*self.lx), -self.ly/60, -17*self.ly/(105*self.lx), -13*self.ly**2/(420*self.lx), -self.ly/60, -46*self.ly/(105*self.lx), 11*self.ly**2/(210*self.lx), -self.ly/30, 46*self.ly/(105*self.lx), -11*self.ly**2/(210*self.lx), -self.ly/30], [-13*self.ly**2/(420*self.lx), -self.ly**3/(140*self.lx), 0, 13*self.ly**2/(420*self.lx), self.ly**3/(140*self.lx), 0, 11*self.ly**2/(210*self.lx), -self.ly**3/(105*self.lx), 0, -11*self.ly**2/(210*self.lx), self.ly**3/(105*self.lx), 0], [-self.ly/60, 0, self.lx*self.ly/45, self.ly/60, 0, -self.lx*self.ly/180, self.ly/30, 0, -self.lx*self.ly/90, -self.ly/30, 0, 2*self.lx*self.ly/45]])
    K_nl_0_Nyy = np.array([[46*self.lx/(105*self.ly), self.lx/30, -11*self.lx**2/(210*self.ly), 17*self.lx/(105*self.ly), self.lx/60, 13*self.lx**2/(420*self.ly), -17*self.lx/(105*self.ly), self.lx/60, -13*self.lx**2/(420*self.ly), -46*self.lx/(105*self.ly), self.lx/30, 11*self.lx**2/(210*self.ly)], [self.lx/30, 2*self.lx*self.ly/45, 0, self.lx/60, self.lx*self.ly/45, 0, -self.lx/60, -self.lx*self.ly/180, 0, -self.lx/30, -self.lx*self.ly/90, 0], [-11*self.lx**2/(210*self.ly), 0, self.lx**3/(105*self.ly), -13*self.lx**2/(420*self.ly), 0, -self.lx**3/(140*self.ly), 13*self.lx**2/(420*self.ly), 0, self.lx**3/(140*self.ly), 11*self.lx**2/(210*self.ly), 0, -self.lx**3/(105*self.ly)], [17*self.lx/(105*self.ly), self.lx/60, -13*self.lx**2/(420*self.ly), 46*self.lx/(105*self.ly), self.lx/30, 11*self.lx**2/(210*self.ly), -46*self.lx/(105*self.ly), self.lx/30, -11*self.lx**2/(210*self.ly), -17*self.lx/(105*self.ly), self.lx/60, 13*self.lx**2/(420*self.ly)], [self.lx/60, self.lx*self.ly/45, 0, self.lx/30, 2*self.lx*self.ly/45, 0, -self.lx/30, -self.lx*self.ly/90, 0, -self.lx/60, -self.lx*self.ly/180, 0], [13*self.lx**2/(420*self.ly), 0, -self.lx**3/(140*self.ly), 11*self.lx**2/(210*self.ly), 0, self.lx**3/(105*self.ly), -11*self.lx**2/(210*self.ly), 0, -self.lx**3/(105*self.ly), -13*self.lx**2/(420*self.ly), 0, self.lx**3/(140*self.ly)], [-17*self.lx/(105*self.ly), -self.lx/60, 13*self.lx**2/(420*self.ly), -46*self.lx/(105*self.ly), -self.lx/30, -11*self.lx**2/(210*self.ly), 46*self.lx/(105*self.ly), -self.lx/30, 11*self.lx**2/(210*self.ly), 17*self.lx/(105*self.ly), -self.lx/60, -13*self.lx**2/(420*self.ly)], [self.lx/60, -self.lx*self.ly/180, 0, self.lx/30, -self.lx*self.ly/90, 0, -self.lx/30, 2*self.lx*self.ly/45, 0, -self.lx/60, self.lx*self.ly/45, 0], [-13*self.lx**2/(420*self.ly), 0, self.lx**3/(140*self.ly), -11*self.lx**2/(210*self.ly), 0, -self.lx**3/(105*self.ly), 11*self.lx**2/(210*self.ly), 0, self.lx**3/(105*self.ly), 13*self.lx**2/(420*self.ly), 0, -self.lx**3/(140*self.ly)], [-46*self.lx/(105*self.ly), -self.lx/30, 11*self.lx**2/(210*self.ly), -17*self.lx/(105*self.ly), -self.lx/60, -13*self.lx**2/(420*self.ly), 17*self.lx/(105*self.ly), -self.lx/60, 13*self.lx**2/(420*self.ly), 46*self.lx/(105*self.ly), -self.lx/30, -11*self.lx**2/(210*self.ly)], [self.lx/30, -self.lx*self.ly/90, 0, self.lx/60, -self.lx*self.ly/180, 0, -self.lx/60, self.lx*self.ly/45, 0, -self.lx/30, 2*self.lx*self.ly/45, 0], [11*self.lx**2/(210*self.ly), 0, -self.lx**3/(105*self.ly), 13*self.lx**2/(420*self.ly), 0, self.lx**3/(140*self.ly), -13*self.lx**2/(420*self.ly), 0, -self.lx**3/(140*self.ly), -11*self.lx**2/(210*self.ly), 0, self.lx**3/(105*self.ly)]])
    K_nl_0_Nxy = np.array([[1/4, -self.ly/20, -self.lx/20, 1/4, -self.ly/20, self.lx/20, -1/4, self.ly/20, -self.lx/20, -1/4, self.ly/20, self.lx/20], [self.ly/20, 0, -self.lx*self.ly/144, self.ly/20, 0, self.lx*self.ly/144, -self.ly/20, self.ly**2/120, -self.lx*self.ly/144, -self.ly/20, self.ly**2/120, self.lx*self.ly/144], [self.lx/20, -self.lx*self.ly/144, 0, -self.lx/20, self.lx*self.ly/144, -self.lx**2/120, self.lx/20, -self.lx*self.ly/144, self.lx**2/120, -self.lx/20, self.lx*self.ly/144, 0], [-1/4, self.ly/20, self.lx/20, -1/4, self.ly/20, -self.lx/20, 1/4, -self.ly/20, self.lx/20, 1/4, -self.ly/20, -self.lx/20], [-self.ly/20, 0, self.lx*self.ly/144, -self.ly/20, 0, -self.lx*self.ly/144, self.ly/20, -self.ly**2/120, self.lx*self.ly/144, self.ly/20, -self.ly**2/120, -self.lx*self.ly/144], [-self.lx/20, self.lx*self.ly/144, self.lx**2/120, self.lx/20, -self.lx*self.ly/144, 0, -self.lx/20, self.lx*self.ly/144, 0, self.lx/20, -self.lx*self.ly/144, -self.lx**2/120], [-1/4, -self.ly/20, self.lx/20, -1/4, -self.ly/20, -self.lx/20, 1/4, self.ly/20, self.lx/20, 1/4, self.ly/20, -self.lx/20], [self.ly/20, self.ly**2/120, -self.lx*self.ly/144, self.ly/20, self.ly**2/120, self.lx*self.ly/144, -self.ly/20, 0, -self.lx*self.ly/144, -self.ly/20, 0, self.lx*self.ly/144], [-self.lx/20, -self.lx*self.ly/144, self.lx**2/120, self.lx/20, self.lx*self.ly/144, 0, -self.lx/20, -self.lx*self.ly/144, 0, self.lx/20, self.lx*self.ly/144, -self.lx**2/120], [1/4, self.ly/20, -self.lx/20, 1/4, self.ly/20, self.lx/20, -1/4, -self.ly/20, -self.lx/20, -1/4, -self.ly/20, self.lx/20], [-self.ly/20, -self.ly**2/120, self.lx*self.ly/144, -self.ly/20, -self.ly**2/120, -self.lx*self.ly/144, self.ly/20, 0, self.lx*self.ly/144, self.ly/20, 0, -self.lx*self.ly/144], [self.lx/20, self.lx*self.ly/144, 0, -self.lx/20, -self.lx*self.ly/144, -self.lx**2/120, self.lx/20, self.lx*self.ly/144, self.lx**2/120, -self.lx/20, -self.lx*self.ly/144, 0]])
    return N_xx*K_nl_0_Nxx + N_yy*K_nl_0_Nyy + N_xy*K_nl_0_Nxy
Thin_Plate_Element.K_nl_0_N_loc_e = K_nl_0_N_loc_e


def Loc_to_Glob_Matr(self, n_dofs):
    T = np.identity(n_dofs)
    TT = T[:, self.dofs_list]
    return TT
Thin_Plate_Element.Loc_to_Glob_Matr = Loc_to_Glob_Matr


def K_lin_glob(self, n_dofs):
    K_glob = self.Loc_to_Glob_Matr(n_dofs)@(self.K_lin_loc_e()@self.Loc_to_Glob_Matr(n_dofs).T)
    return K_glob
Thin_Plate_Element.K_lin_glob = K_lin_glob


def M_glob(self, n_dofs):
    M_glob = self.Loc_to_Glob_Matr(n_dofs)@(self.M_loc_e()@self.Loc_to_Glob_Matr(n_dofs).T)
    return M_glob
Thin_Plate_Element.M_glob = M_glob

def U_loc_glob(self, U_glob):
    return U_glob[self.dofs_list]
Thin_Plate_Element.U_loc_glob = U_loc_glob

def deflections(self, u, numnum):
    xe, ye = np.linspace(self.x0, self.x0+self.lx, num = numnum), np.linspace(self.y0, self.y0+self.ly, num = numnum)
    x, y = np.meshgrid(xe, ye)
    N = np.array([1 - 3*(y-self.y0)**2/self.ly**2 + 2*(y-self.y0)**3/self.ly**3 - (x-self.x0)*(y-self.y0)/(self.lx*self.ly) + 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) - 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) - 3*(x-self.x0)**2/self.lx**2 + 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) + 2*(x-self.x0)**3/self.lx**3 - 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    (y-self.y0) - 2*(y-self.y0)**2/self.ly + (y-self.y0)**3/self.ly**2 - (x-self.x0)*(y-self.y0)/self.lx + 2*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) - (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    -(x-self.x0) + (x-self.x0)*(y-self.y0)/self.ly + 2*(x-self.x0)**2/self.lx - 2*(x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3/self.lx**2 + (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    (x-self.x0)*(y-self.y0)/(self.lx*self.ly) - 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) + 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) + 3*(x-self.x0)**2/self.lx**2 - 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) - 2*(x-self.x0)**3/self.lx**3 + 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    (x-self.x0)*(y-self.y0)/self.lx - 2*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) + (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    (x-self.x0)**2/self.lx - (x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3/self.lx**2 + (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    -(x-self.x0)*(y-self.y0)/(self.lx*self.ly) + 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) - 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) + 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) - 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    -(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) + (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    (x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    3*(y-self.y0)**2/self.ly**2 - 2*(y-self.y0)**3/self.ly**3 + (x-self.x0)*(y-self.y0)/(self.lx*self.ly) - 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) + 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) - 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) + 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    -(y-self.y0)**2/self.ly + (y-self.y0)**3/self.ly**2 + (x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) - (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    -(x-self.x0)*(y-self.y0)/self.ly + 2*(x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly)
    ])
    w = np.einsum('ijk, i', N, u)
    return x, y, w
Thin_Plate_Element.deflections = deflections

def MxValues(self, u, numnum):
    xe, ye = np.linspace(self.x0, self.x0+self.lx, num = numnum), np.linspace(self.y0, self.y0+self.ly, num = numnum)
    x, y = np.meshgrid(xe, ye)
    d2Ndx2 = np.array([6*(-1 + (y-self.y0)/self.ly + 2*(x-self.x0)/self.lx - 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx**2,
                0*x*y,
                2*(2 - 2*(y-self.y0)/self.ly - 3*(x-self.x0)/self.lx + 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx,
                6*(1 - (y-self.y0)/self.ly - 2*(x-self.x0)/self.lx + 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx**2,
                0*x*y,
                2*(1 - (y-self.y0)/self.ly - 3*(x-self.x0)/self.lx + 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx,
                6*(y-self.y0)*(1 - 2*(x-self.x0)/self.lx)/(self.lx**2*self.ly),
                0*x*y,
                2*(y-self.y0)*(1 - 3*(x-self.x0)/self.lx)/(self.lx*self.ly),
                6*(y-self.y0)*(-1 + 2*(x-self.x0)/self.lx)/(self.lx**2*self.ly),
                0*x*y,
                2*(y-self.y0)*(2 - 3*(x-self.x0)/self.lx)/(self.lx*self.ly)
                ])
    d2Ndy2 = np.array([6*(-1 + 2*(y-self.y0)/self.ly + (x-self.x0)/self.lx - 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly**2,
            2*(-2 + 3*(y-self.y0)/self.ly + 2*(x-self.x0)/self.lx - 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly,
            0*x*y,
            6*(x-self.x0)*(-1 + 2*(y-self.y0)/self.ly)/(self.lx*self.ly**2),
            2*(x-self.x0)*(-2 + 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            0*x*y,
            6*(x-self.x0)*(1 - 2*(y-self.y0)/self.ly)/(self.lx*self.ly**2),
            2*(x-self.x0)*(-1 + 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            0*x*y,
            6*(1 - 2*(y-self.y0)/self.ly - (x-self.x0)/self.lx + 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly**2,
            2*(-1 + 3*(y-self.y0)/self.ly + (x-self.x0)/self.lx - 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly,
            0*x*y])
    Mx = -self.D * (np.einsum('ijk, i', d2Ndx2, u)+self.nu*np.einsum('ijk, i', d2Ndy2, u))
    return x, y, Mx
Thin_Plate_Element.MxValues = MxValues

def MyValues(self, u, numnum):
    xe, ye = np.linspace(self.x0, self.x0+self.lx, num = numnum), np.linspace(self.y0, self.y0+self.ly, num = numnum)
    x, y = np.meshgrid(xe, ye)
    d2Ndx2 = np.array([6*(-1 + (y-self.y0)/self.ly + 2*(x-self.x0)/self.lx - 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx**2,
                0*x*y,
                2*(2 - 2*(y-self.y0)/self.ly - 3*(x-self.x0)/self.lx + 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx,
                6*(1 - (y-self.y0)/self.ly - 2*(x-self.x0)/self.lx + 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx**2,
                0*x*y,
                2*(1 - (y-self.y0)/self.ly - 3*(x-self.x0)/self.lx + 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.lx,
                6*(y-self.y0)*(1 - 2*(x-self.x0)/self.lx)/(self.lx**2*self.ly),
                0*x*y,
                2*(y-self.y0)*(1 - 3*(x-self.x0)/self.lx)/(self.lx*self.ly),
                6*(y-self.y0)*(-1 + 2*(x-self.x0)/self.lx)/(self.lx**2*self.ly),
                0*x*y,
                2*(y-self.y0)*(2 - 3*(x-self.x0)/self.lx)/(self.lx*self.ly)
                ])
    d2Ndy2 = np.array([6*(-1 + 2*(y-self.y0)/self.ly + (x-self.x0)/self.lx - 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly**2,
            2*(-2 + 3*(y-self.y0)/self.ly + 2*(x-self.x0)/self.lx - 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly,
            0*x*y,
            6*(x-self.x0)*(-1 + 2*(y-self.y0)/self.ly)/(self.lx*self.ly**2),
            2*(x-self.x0)*(-2 + 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            0*x*y,
            6*(x-self.x0)*(1 - 2*(y-self.y0)/self.ly)/(self.lx*self.ly**2),
            2*(x-self.x0)*(-1 + 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            0*x*y,
            6*(1 - 2*(y-self.y0)/self.ly - (x-self.x0)/self.lx + 2*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly**2,
            2*(-1 + 3*(y-self.y0)/self.ly + (x-self.x0)/self.lx - 3*(x-self.x0)*(y-self.y0)/(self.lx*self.ly))/self.ly,
            0*x*y])
    My = -self.D * (self.nu*np.einsum('ijk, i', d2Ndx2, u)+np.einsum('ijk, i', d2Ndy2, u))
    return x, y, My
Thin_Plate_Element.MyValues = MyValues

def MxyValues(self, u, numnum):
    xe, ye = np.linspace(self.x0, self.x0+self.lx, num = numnum), np.linspace(self.y0, self.y0+self.ly, num = numnum)
    x, y = np.meshgrid(xe, ye)
    d2Ndxdy = np.array([(-1 + 6*(y-self.y0)/self.ly - 6*(y-self.y0)**2/self.ly**2 + 6*(x-self.x0)/self.lx - 6*(x-self.x0)**2/self.lx**2)/(self.lx*self.ly),
            (-1 + 4*(y-self.y0)/self.ly - 3*(y-self.y0)**2/self.ly**2)/self.lx,
            (1 - 4*(x-self.x0)/self.lx + 3*(x-self.x0)**2/self.lx**2)/self.ly,
            (1 - 6*(y-self.y0)/self.ly + 6*(y-self.y0)**2/self.ly**2 - 6*(x-self.x0)/self.lx + 6*(x-self.x0)**2/self.lx**2)/(self.lx*self.ly),
            (1 - 4*(y-self.y0)/self.ly + 3*(y-self.y0)**2/self.ly**2)/self.lx,
            (x-self.x0)*(-2 + 3*(x-self.x0)/self.lx)/(self.lx*self.ly),
            (-1 + 6*(y-self.y0)/self.ly - 6*(y-self.y0)**2/self.ly**2 + 6*(x-self.x0)/self.lx - 6*(x-self.x0)**2/self.lx**2)/(self.lx*self.ly),
            (y-self.y0)*(-2 + 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            (x-self.x0)*(2 - 3*(x-self.x0)/self.lx)/(self.lx*self.ly),
            (1 - 6*(y-self.y0)/self.ly + 6*(y-self.y0)**2/self.ly**2 - 6*(x-self.x0)/self.lx + 6*(x-self.x0)**2/self.lx**2)/(self.lx*self.ly),
            (y-self.y0)*(2 - 3*(y-self.y0)/self.ly)/(self.lx*self.ly),
            (-1 + 4*(x-self.x0)/self.lx - 3*(x-self.x0)**2/self.lx**2)/self.ly
            ])
    Mxy = -self.D * ((1-self.nu)*np.einsum('ijk, i', d2Ndxdy, u))
    return x, y, Mxy
Thin_Plate_Element.MxyValues = MxyValues

def Shape_Fcs(self, x, y):
    N = [1 - 3*(y-self.y0)**2/self.ly**2 + 2*(y-self.y0)**3/self.ly**3 - (x-self.x0)*(y-self.y0)/(self.lx*self.ly) + 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) - 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) - 3*(x-self.x0)**2/self.lx**2 + 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) + 2*(x-self.x0)**3/self.lx**3 - 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    (y-self.y0) - 2*(y-self.y0)**2/self.ly + (y-self.y0)**3/self.ly**2 - (x-self.x0)*(y-self.y0)/self.lx + 2*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) - (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    -(x-self.x0) + (x-self.x0)*(y-self.y0)/self.ly + 2*(x-self.x0)**2/self.lx - 2*(x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3/self.lx**2 + (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    (x-self.x0)*(y-self.y0)/(self.lx*self.ly) - 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) + 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) + 3*(x-self.x0)**2/self.lx**2 - 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) - 2*(x-self.x0)**3/self.lx**3 + 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    (x-self.x0)*(y-self.y0)/self.lx - 2*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) + (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    (x-self.x0)**2/self.lx - (x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3/self.lx**2 + (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    -(x-self.x0)*(y-self.y0)/(self.lx*self.ly) + 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) - 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) + 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) - 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    -(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) + (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    (x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly),
    3*(y-self.y0)**2/self.ly**2 - 2*(y-self.y0)**3/self.ly**3 + (x-self.x0)*(y-self.y0)/(self.lx*self.ly) - 3*(x-self.x0)*(y-self.y0)**2/(self.lx*self.ly**2) + 2*(x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**3) - 3*(x-self.x0)**2*(y-self.y0)/(self.lx**2*self.ly) + 2*(x-self.x0)**3*(y-self.y0)/(self.lx**3*self.ly),
    -(y-self.y0)**2/self.ly + (y-self.y0)**3/self.ly**2 + (x-self.x0)*(y-self.y0)**2/(self.lx*self.ly) - (x-self.x0)*(y-self.y0)**3/(self.lx*self.ly**2),
    -(x-self.x0)*(y-self.y0)/self.ly + 2*(x-self.x0)**2*(y-self.y0)/(self.lx*self.ly) - (x-self.x0)**3*(y-self.y0)/(self.lx**2*self.ly)
    ]
    return N
Thin_Plate_Element.Shape_Fcs = Shape_Fcs
