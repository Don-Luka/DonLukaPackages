'''
NOT OPTIMIZED YET
'''

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

# these are new

def display_plot_structure(XY, elems, plot_scale=5):
    n_nodes, n_elems = np.size(XY, axis=0), np.size(elems, axis=0)
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    for j in range(n_elems):
        tt = np.vstack((XY[elems[j]][:,0], XY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'grid')
    plt.show()
    
def save_plot_structure(XY, elems, total_path, plot_scale=5):
    n_nodes, n_elems = np.size(XY, axis=0), np.size(elems, axis=0)
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    for j in range(n_elems):
        tt = np.vstack((XY[elems[j]][:,0], XY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'grid')
    fig_name = f'structure' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=200)
    plt.close()

def display_displacements(elems, XY, U_glob, plot_scale=5, scale_factor=1):
    n_nodes, n_elems = np.size(XY, axis=0), np.size(elems, axis=0)
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    UV = scale_factor*np.reshape(U_glob,(np.size(XY, axis=0),-1))/np.max(np.abs(U_glob))
    XYN = XY+UV
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    for j in range(n_elems):
        tt = np.vstack((XY[elems[j]][:,0], XY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k',facecolor='gray') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    for j in range(n_elems):
        tt = np.vstack((XYN[elems[j]][:,0], XYN[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k',facecolor='r') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'grid')
    plt.show()

def save_displacements(elems, XY, U_glob, total_path, plot_suffix = None, plot_scale=5, scale_factor=1):
    n_nodes, n_elems = np.size(XY, axis=0), np.size(elems, axis=0)
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    UV = scale_factor*np.reshape(U_glob,(np.size(XY, axis=0),-1))/np.max(np.abs(U_glob))
    XYN = XY+UV
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    for j in range(n_elems):
        tt = np.vstack((XY[elems[j]][:,0], XY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k',facecolor='gray') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    for j in range(n_elems):
        tt = np.vstack((XYN[elems[j]][:,0], XYN[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k',facecolor='r') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'grid')
    if plot_suffix == None:
        fig_name = f'deflections' # add suffix
    else:
        fig_name = f'deflections {plot_suffix}' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()

def display_arf_plot(freq_vec, eig_freqs, Amp):
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(12, 6))
    ax.plot(freq_vec, Amp, c='k', linewidth=2, label = "deflection")
    for j in range(np.size(eig_freqs)):
        plt.axvline(eig_freqs[j], linestyle='--', c='k', linewidth=1)
    ax.legend()
    ax.autoscale(enable=True, axis='both', tight=False)
    fig.suptitle("amplitude response")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_ylabel('$w(f)$')
    plt.pause(5)
    plt.show()

def save_arf_plot(freq_vec, eig_freqs, Amp, total_path):
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(12, 6))
    ax.plot(freq_vec, Amp, c='k', linewidth=2, label = "deflection")
    for j in range(np.size(eig_freqs)):
        plt.axvline(eig_freqs[j], linestyle='--', c='k', linewidth=1)
    ax.legend()
    ax.autoscale(enable=True, axis='both', tight=False)
    fig.suptitle("amplitude response")
    ax.set_xlabel('$f$ [Hz]')
    ax.set_ylabel('$w(f)$')
    fig_name = f'ARF' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()




# and these are old
    
def display_deflections(XY, UXY, elems, plot_scale=5):
    '''
    This function is withdrawn, do not use it.
    '''
    n_nodes, n_elems = np.size(XY, axis=0), np.size(elems, axis=0)
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    for j in range(n_elems):
        tt = np.vstack((XY[elems[j]][:,0], XY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    for j in range(n_elems):
        tt = np.vstack((UXY[elems[j]][:,0], UXY[elems[j]][:,1])).T
        poly = Polygon(tt,closed=True,edgecolor='k',facecolor='r') # ,facecolor='gray'
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'grid')
    plt.show()



def dofsplots(XX_v_nodes, YY_v_nodes, num_y, scalexy, path_):
    L_x, L_y = np.max(XX_v_nodes)-np.min(XX_v_nodes), np.max(YY_v_nodes)-np.min(YY_v_nodes)
    n_nodes = np.size(XX_v_nodes)
    fig, ax = plt.subplots(2,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,2*scalexy*L_y))
    ax[0].scatter(XX_v_nodes, YY_v_nodes)
    for j in range(n_nodes):
        ax[0].text(XX_v_nodes[j], 
                YY_v_nodes[j]+0.5*L_y/(num_y-1), 
                (j), size=10, ha='center', va='center')    
    ax[0].autoscale(enable=True, axis='both', tight=False)
    ax[0].axis('off')
    ax[1].scatter(XX_v_nodes, YY_v_nodes)
    for j in range(n_nodes):
        ax[1].text(XX_v_nodes[j], 
                YY_v_nodes[j]+0.5*L_y/(num_y-1), 
                (j+n_nodes), size=10, ha='center', va='center')    
    ax[1].autoscale(enable=True, axis='both', tight=False)
    ax[1].axis('off')
    fname = os.path.join(path_,"dofs.png")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def uplots(L_x, L_y, u_stacked, scalexy, path_):
    fname = os.path.join(path_,"deflections.png")
    fig, ax = plt.subplots(2,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,2*scalexy*L_y))
    ax[0].contourf(u_stacked[0],u_stacked[1],u_stacked[2])
    ax[0].autoscale(enable=True, axis='both', tight=False)
    ax[0].set_title('$u_x$')
    ax[1].contourf(u_stacked[0],u_stacked[1],u_stacked[3])
    ax[1].autoscale(enable=True, axis='both', tight=False)
    ax[1].set_title('$u_y$')
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close()

def stressplots(L_x, L_y, sigma_stacked, scalexy, path_):
    fname = os.path.join(path_,"stress.png")
    fig, ax = plt.subplots(3,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,3*scalexy*L_y))
    ax[0].contourf(sigma_stacked[0],sigma_stacked[1],sigma_stacked[2])
    ax[0].autoscale(enable=True, axis='both', tight=False)
    ax[0].set_title('$\sigma_x$')
    ax[1].contourf(sigma_stacked[0],sigma_stacked[1],sigma_stacked[3])
    ax[1].autoscale(enable=True, axis='both', tight=False)
    ax[1].set_title('$\sigma_y$')
    ax[2].contourf(sigma_stacked[0],sigma_stacked[1],sigma_stacked[4])
    ax[2].autoscale(enable=True, axis='both', tight=False)
    ax[2].set_title('$\tau_{xy}$')
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close()

    fname = os.path.join(path_,"principal_stress.png")
    fig, ax = plt.subplots(2,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,3*scalexy*L_y))
    ax[0].contourf(sigma_stacked[0],sigma_stacked[1],sigma_stacked[5])
    ax[0].autoscale(enable=True, axis='both', tight=False)
    ax[0].set_title('$\sigma_1$')
    ax[1].contourf(sigma_stacked[0],sigma_stacked[1],sigma_stacked[6])
    ax[1].autoscale(enable=True, axis='both', tight=False)
    ax[1].set_title('$\sigma_2$')
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close()

def plotsdefl(L_x, L_y, scalexy, scalef, n_elems, x_nodes_e, y_nodes_e, u_loc, v_loc, path_):
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,scalexy*L_y))
    for j in range(n_elems):
        ss = np.vstack((x_nodes_e[j]+scalef*u_loc[j], y_nodes_e[j]+scalef*v_loc[j])).T
        ss[[-2,-1],:] = ss[[-1,-2],:]
        poly = Polygon(ss,closed=True,color='b', alpha=0.2)
        plt.gca().add_patch(poly)
        tt = np.vstack((x_nodes_e[j], y_nodes_e[j])).T
        tt[[-2,-1],:] = tt[[-1,-2],:]
        poly = Polygon(tt,closed=True,color='r', alpha=0.1)
    for j in range(n_elems):
        ax.scatter(x_nodes_e[j]+scalef*u_loc[j], y_nodes_e[j]+scalef*v_loc[j],c='r')
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    fname = os.path.join(path_,"defl.png")
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close

def plotsdefl_neat(L_x, L_y, num_x, num_y, XY_v_nodes, n_nodes, scalexy, smoothness, U_glob, path_):
    X_nodes1, Y_nodes1 = np.linspace(0,L_x, num = smoothness*num_x), np.linspace(0,L_y, num = smoothness*num_y)
    XX_nodes1, YY_nodes1 = np.meshgrid(X_nodes1,Y_nodes1)
    grid_z_u = griddata(XY_v_nodes, U_glob[:n_nodes], (XX_nodes1, YY_nodes1), method='cubic')
    grid_z_v = griddata(XY_v_nodes, U_glob[n_nodes:2*n_nodes], (XX_nodes1, YY_nodes1), method='cubic')
    fig, ax = plt.subplots(2,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,2*scalexy*L_y))
    ax[0].imshow(grid_z_u, extent=(0,L_x,0,L_y))
    ax[0].autoscale(enable=True, axis='both', tight=False)
    ax[1].imshow(grid_z_v, extent=(0,L_x,0,L_y))
    ax[1].autoscale(enable=True, axis='both', tight=False)
    fname = os.path.join(path_,"defl_smoth.png")
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close()

def plotsgridelm(L_x, L_y, scalexy, n_elems, x_nodes_e, y_nodes_e, path_):
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,scalexy*L_y))
    for j in range(n_elems):
        tt = np.vstack((x_nodes_e[j], y_nodes_e[j])).T
        tt[[-2,-1],:] = tt[[-1,-2],:]
        poly = Polygon(tt,closed=True,color=str(1-j/n_elems))
        plt.gca().add_patch(poly)
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    fname = os.path.join(path_,"elemgrid.png")
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close()

def plotsdeflevect(L_x, L_y, scalexy, scalef, n_elems, x_nodes_e, y_nodes_e, u_loc_, v_loc_, path_, ordnum):
    u_loc, v_loc = u_loc_[ordnum], v_loc_[ordnum]
    suffix_2 = str(ordnum)
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(scalexy*L_x,scalexy*L_y))
    for j in range(n_elems):
        ss = np.vstack((x_nodes_e[j]+scalef*u_loc[j], y_nodes_e[j]+scalef*v_loc[j])).T
        ss[[-2,-1],:] = ss[[-1,-2],:]
        poly = Polygon(ss,closed=True,color='b', alpha=0.2)
        plt.gca().add_patch(poly)
        tt = np.vstack((x_nodes_e[j], y_nodes_e[j])).T
        tt[[-2,-1],:] = tt[[-1,-2],:]
        poly = Polygon(tt,closed=True,color='r', alpha=0.1)
    for j in range(n_elems):
        ax.scatter(x_nodes_e[j]+scalef*u_loc[j], y_nodes_e[j]+scalef*v_loc[j],c='r')
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    fname = os.path.join(path_,"mode"+suffix_2+"defl.png")
    plt.savefig(fname, bbox_inches='tight',dpi=150)
    plt.close









