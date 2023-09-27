import numpy as np
import matplotlib.pyplot as plt

def plot_aux_structure(XY, elems, total_path):
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
    plt.pause(5)
    plt.close()

def plot_colored_aux_structure(XY, elems, total_path):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c=(0.5*j/np.size(elems,axis=0),0.1*j/np.size(elems,axis=0),2*0.5*j/np.size(elems,axis=0)),linewidth=3)
    fig_name = f'structure' # add suffix
    plt.savefig(f'{total_path}\{fig_name}.png', dpi=100)
    plt.pause(5)
    plt.close()

def display_colored_aux_structure(XY, elems, colorsj):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.set_aspect('equal','box')
    # ax.scatter(XY[:,0], XY[:,1], c='k', s=30)
    for j in range(np.size(elems,axis=0)):
        n1, n2 = elems[j][0],elems[j][1]
        ax.plot(
                np.array([XY[n1,0], XY[n2,0]]),
                np.array([XY[n1,1], XY[n2,1]])
        ,c=(colorsj[j],0.5*colorsj[j],1-colorsj[j]),linewidth=3)
    plt.pause(15)
    plt.close()

