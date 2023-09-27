import sys
import numpy as np
import matplotlib.pyplot as plt

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

def display_plot_structure(XY, elems):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='k',linewidth=3)
    plt.pause(5)
    plt.close()

def save_plot_structure(XY, elems, total_path):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='k',linewidth=3)
    fig_name = f'structure' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()

def save_plot_structure_with_suffix(XY, elems, total_path, plot_suffix = None):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c='k',linewidth=3)
    if plot_suffix == None:
        fig_name = f'structure' # add suffix
    else:
        fig_name = f'structure {plot_suffix}' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()

def display_displacements(elems, XY, U_glob, scale_factor=1):
    UV = np.reshape(U_glob,(np.size(XY, axis=0),-1))[:,:-1]
    UV = scale_factor*UV/np.max(np.abs(UV))
    XYN = XY+UV
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
    ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c='r',linewidth=3)
    plt.pause(30)
    plt.close()
    
def save_displacements(elems, XY, U_glob, total_path, plot_suffix = None, scale_factor=1):
    UV = np.reshape(U_glob,(np.size(XY, axis=0),-1))[:,:-1]
    UV = scale_factor*UV/np.max(np.abs(UV))
    XYN = XY+UV
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
    ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c='r',linewidth=3)
    # plt.pause(15)
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

# The rest is thrash
    
    
    
    
    
    
def display_statics_problem_O(elems, XY, XYN):
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
    ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c='r',linewidth=3)
    plt.pause(5)
    plt.close()

def display_statics_problem_colored_O(elems, XY, XYN, colorsj):
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

def save_statics_problem_O(elems, XY, XYN, total_path):
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
    ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c='r',linewidth=3)
    # plt.pause(15)
    fig_name = f'deflections' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()
    
def save_statics_problem_colored_O(elems, XY, XYN, colorsj, total_path):
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
    
def save_eigenmode_O(elems, XY, XYN, total_path, mode_number):
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
    ax.scatter(XYN[:,0], XYN[:,1], c='r', s=30)
    for j in range(n_elems):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XYN[n1,0], XYN[n2,0]]),
                np.array([XYN[n1,1], XYN[n2,1]])
        ,c='r',linewidth=3)
    # plt.pause(15)
    fig_name = f'mode{mode_number}' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.close()