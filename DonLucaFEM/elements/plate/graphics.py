import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.pyplot import fill, fill_between
from matplotlib import cm  # color map
import scipy

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
    plt.pause(5)
    plt.close()

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

def display_deflections_3D(elements, U_glob, precision=2):
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection='3d')
    for elem in elements:
        xx, yy, ww = elem.deflections(elem.U_loc_glob(U_glob), precision)
        ax.plot_surface(xx, yy, ww,shade=True, color='gray') # ,color='gray'
    ax.azim = -15
    # ax.dist = 10
    ax.elev = 30
    ax.autoscale(enable=True,tight=True)
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$W$')
    fig.suptitle(f'deflection3d')
    plt.pause(15)
    plt.close()

def save_deflections_3D(elements, U_glob, total_path, plot_suffix = None, precision=2):
    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection='3d')
    for elem in elements:
        xx, yy, ww = elem.deflections(elem.U_loc_glob(U_glob), precision)
        ax.plot_surface(xx, yy, ww,shade=True, color='gray') # ,color='gray'
    ax.azim = -60
    # ax.dist = 10
    ax.elev = 30
    ax.autoscale(enable=True,tight=True)
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$W$')
    fig.suptitle(f'deflection3d')
    if plot_suffix == None:
        fig_name = f'deflections 3D' # add suffix
    else:
        fig_name = f'deflections 3D {plot_suffix}' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=200)
    plt.close()

def display_simple_deflections_2D(XY, W_glob, interp_degree=3, plot_scale=5):
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    uniqueX, uniqueY = np.unique(XY[:,0]),np.unique(XY[:,1])
    uniX, uniY = np.meshgrid(uniqueX, uniqueY)
    uniW = np.reshape(W_glob,(np.size(uniqueY),np.size(uniqueX)))
    UniX, UniY, UniW = scipy.ndimage.zoom(uniX,interp_degree), scipy.ndimage.zoom(uniY,interp_degree), scipy.ndimage.zoom(uniW,interp_degree)
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    plt.contourf(UniX, UniY, UniW, cmap='binary')
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'deflection')
    plt.pause(5)
    plt.close()
    
def save_simple_deflections_2D(XY, W_glob, total_path, plot_suffix = None, interp_degree=3, plot_scale=5):
    Lx = np.max(XY[:,0])-np.min(XY[:,0])
    Ly = np.max(XY[:,1])-np.min(XY[:,1])
    uniqueX, uniqueY = np.unique(XY[:,0]),np.unique(XY[:,1])
    uniX, uniY = np.meshgrid(uniqueX, uniqueY)
    uniW = np.reshape(W_glob,(np.size(uniqueY),np.size(uniqueX)))
    UniX, UniY, UniW = scipy.ndimage.zoom(uniX,interp_degree), scipy.ndimage.zoom(uniY,interp_degree), scipy.ndimage.zoom(uniW,interp_degree)
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(plot_scale*Lx,plot_scale*Ly))
    plt.contourf(UniX, UniY, UniW, cmap='RdBu_r')
    ax.axis('equal')
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    fig.suptitle(f'deflection')
    if plot_suffix == None:
        fig_name = f'deflections 2D' # add suffix
    else:
        fig_name = f'deflections 2D {plot_suffix}' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=200)
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









if __name__ == '__main__':
    sys.exit()



def display_plot_structure_thickness(XY, elems, thickness):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='k',linewidth=thickness[j])
    plt.pause(5)
    plt.close()


def display_statics_problem_colored(elems, XY, XYN, colorsj):
    fig, ax = plt.subplots()
    ax.set_aspect('equal','box')
    ax.set_facecolor('black')
    # ax.scatter(XY[:,0], XY[:,1], c='gray')
    n_elems = np.size(elems,axis=0)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='gray')
    # ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c=(colorsj[j],0,1-colorsj[j]),linewidth=3)
    plt.pause(5)
    plt.close()
    
def save_statics_problem_colored(elems, XY, XYN, colorsj, total_path):
    fig, ax = plt.subplots()
    ax.set_aspect('equal','box')
    ax.set_facecolor('black')
    # ax.scatter(XY[:,0], XY[:,1], c='gray')
    n_elems = np.size(elems,axis=0)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='gray')
    # ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c=(colorsj[j],1/2*colorsj[j],1-colorsj[j]),linewidth=3)
    fig_name = f'deflections_with_N' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()
    