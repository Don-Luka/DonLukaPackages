import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import itertools

import random

import time
random.seed(time.time())

# blend crossover
def BLXa(A, B, hh_min, hh_max, alpha):
    '''blends vectors A and B'''
    d = np.abs(A-B)
    e_ = np.size(A)
    u, v = np.empty(e_), np.empty(e_)
    for i in range(e_):
        u[i] = random.uniform(
            max(min(A[i],B[i])-alpha*d[i],hh_min),
            min(max(A[i],B[i])+alpha*d[i],hh_max)
            )
        v[i] = random.uniform(
            max(min(A[i],B[i])-alpha*d[i],hh_min),
            min(max(A[i],B[i])+alpha*d[i],hh_max)
            )
    C = np.stack((u,v),axis=1)
    return C

# one-point crossover
def OPCX(A, B):
    e_ = np.size(A)
    c1 = random.randint(1,e_-1)
    child1 = np.empty(e_)
    child2 = np.empty(e_)
    child1 = np.concatenate((A[:c1],B[c1:]),axis=None)
    child2 = np.concatenate((B[:c1],A[c1:]),axis=None)
    C = np.stack((child1,child2),axis=1)
    return C

# two-point crossover
def TPCX(A, B):
    e_ = np.size(A)
    cp1 = random.randint(1,e_-1)
    cp2 = random.randint(2,e_-1)
    C1, C2 = min(cp1, cp2), max(cp1, cp2)
    child1 = np.empty(e_)
    child2 = np.empty(e_)
    child1 = np.concatenate((A[:C1],B[C1:C2],A[C2:]),axis=None)
    child2 = np.concatenate((B[:C1],A[C1:C2],B[C2:]),axis=None)
    C = np.stack((child1,child2),axis=1)
    return C

# random-points crossover (NEW)
def RPCX(A, B):
    e_ = np.size(A)
    cps = np.random.randint(0,2, e_)
    child1 = A*cps+B*(1-cps)
    child2 = A*(1-cps)+B*cps
    C = np.stack((child1,child2),axis=1)
    return C

# non-uniform mutation
def NU_Mutation(X, z_, T_, b_, hh_min, hh_max):
    '''
    xi_ - mutation type probability 0/1
    z_ - population clock ticking - number of actual population
    r_ - mutation rate coefficient (random from 0.0 to 1.0)
    b_ - still (2)
    '''
    e_ = np.size(X)
    xi_ = random.randint(0,1)
    row_ = random.randint(0,e_-1)
    r_ = random.random()
    X_=np.copy(X)
    if xi_==0:
        X_[row_]=X_[row_]+(hh_max-X_[row_])*(1-r_**((1-z_/T_)**b_))
    else:
        X_[row_]=X_[row_]-(X_[row_]-hh_min)*(1-r_**((1-z_/T_)**b_))
    return X_

def rankings(H_, obj_func, numfreq): # may be changed - incremental, decremental
    """
    THIS FUNCTION IS CHANGED
    """
    popul_size = np.size(H_, axis=1)
    OF = np.asarray([obj_func(numfreq,H_[:,j]) for j in range(popul_size)])
    # idx = OF.argsort()[::-1] # if maximizing
    idx = OF.argsort()[::-1]   
    OF = OF[idx]
    H = H_[:,idx]
    # partial probability
    pprob = OF/np.sum(OF) # here may be some differences in older versions of this procedure (nevermind)
    # cumulative probability
    cumpprob = np.asarray([np.sum(pprob[:x+1]/np.sum(pprob)) for x in range(np.size(OF))])
    return OF, H, pprob, cumpprob

def n_par_calc(popul_size, n_pop, n_left, n_off_from_cross):
    '''Calculates the number of parents needed to create next population'''
    n_par = int(
        np.ceil(1/2*(1+np.sqrt(1+8*(popul_size-n_left)/n_off_from_cross))
            )
        )
    while n_off_from_cross*1/2*n_par*(n_par-1)+n_left -n_pop < 0:
        n_par = n_par+1
    return n_par


def plot_popul(H, popul_number, path, id_):
    if popul_number>=100:
        filename_core = (f'{id_}_H_popul_{popul_number}.png')
    else:
        if popul_number>=10:
            filename_core = (f'{id_}_H_popul_0{popul_number}.png')
        else:
            filename_core = (f'{id_}_H_popul_00{popul_number}.png')
    popul_size, n_elems = np.size(H, axis=1), np.size(H, axis=0)
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(8, 4))
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.imshow(H, cmap = 'seismic')
    fig.suptitle(f'population number {popul_number}')
    ax.set_xlabel('individuals')
    ax.set_ylabel('elements')
    plt.savefig(f'{path}\{filename_core}',dpi=150)
    plt.close()

def plot_OF_history(OF_history, path, id_):
    filename_core = (f'{id_}_OF_history.png')
    fig, ax = plt.subplots(facecolor=(1, 1, 1), figsize=(12, 6))
    ax.plot(np.arange(np.size(OF_history,axis=0)),OF_history[:,0],color='r', linewidth=3)
    ax.autoscale(enable=True, axis='both', tight=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("objective function history")
    ax.set_xlabel('population number $n$')
    ax.set_ylabel('$f_{max}(n)$')
    plt.savefig(f'{path}\{filename_core}',dpi=150)
    plt.close()

def plot_best(H, popul_number, path, id_):
    if popul_number>=100:
        filename_core = (f'{id_}_best_popul_{popul_number}.png')
    else:
        if popul_number>=10:
            filename_core = (f'{id_}_best_popul_0{popul_number}.png')
        else:
            filename_core = (f'{id_}_best_popul_00{popul_number}.png')
    n_elems = np.size(H, axis=0)
    fig, ax = plt.subplots(1,1,facecolor=(1, 1, 1), figsize=(12, 3))
    ax.plot(np.arange(np.size(H,axis=0))/(n_elems-1),-1/2*H[:,0],color='k', linewidth=3)
    ax.plot(np.arange(np.size(H,axis=0))/(n_elems-1),1/2*H[:,0],color='k', linewidth=3)
    ax.fill_between(np.arange(np.size(H,axis=0))/(n_elems-1),-1/2*H[:,0],1/2*H[:,0],color='gray')
    ax.autoscale(enable=True, axis='both', tight=False)
    fig.suptitle(f'best solution population {popul_number}')
    ax.set_xlabel('$x/L$')
    ax.set_ylabel('$h(x)$')
    plt.savefig(f'{path}\{filename_core}',dpi=150)
    plt.close()




def Genetics(popul_size, n_pop, n_left, n_off_from_cross, mrc, n_par, obj_func, 
                numfreq, h_min, h_max, H, z):
    
    # not of any use here, but may be later
    mutation_probability = 0.5
    
    # z is the pseudo-time - number of actual generation
    # in more general cases numfreq doesn't have to be specifierd here
        
    n_elems = np.size(H, axis=0)
    
    # mutation of all the left individuals except the best one
    # may be discussed
    for j in range(1,n_left+1):
        col_ = j
        candidate = NU_Mutation(H[:,col_], z, n_pop, mrc, h_min, h_max)
        if obj_func(numfreq,candidate) >= obj_func(numfreq,H[:,col_]):
            H[:,col_] = candidate
        else:
            pass

    # for j in range(int(mutation_probability*popul_size)):
    #     col_ = random.choice(list(np.arange(popul_size)))
    #     H[:,col_] = NU_Mutation(H[:,col_], z, n_pop, mrc, h_min, h_max)

    # col_ = random.choice(list(np.arange(1,popul_size)))
    # H[:,col_] = NU_Mutation(H[:,col_], z, n_pop, mrc, h_min, h_max)

    # ranking calculation
    OF, H, pprob, cumpprob = rankings(H, obj_func, numfreq)

    # initializing new set of parents
    set_parents = []
    # drawing parents (rhoullette wheel!) until the required number of parents is reached
    # while len(set_parents)<n_par:
    #     index_ = np.where(cumpprob > random.random())[0][0]
    #     set_parents.append(index_)
    #     set_parents = list(set(set_parents))
    while len(set_parents)<n_par:
        if len(set_parents) <= int(n_par/2):
            index_ = np.where(cumpprob > random.random())[0][0]
        else:
            index_ = random.choice(list(np.arange(popul_size)))
        set_parents.append(index_)
        set_parents = list(set(set_parents))
    # creating (all possible) pairs of (numbers of) drawn parents
    pairs = np.random.permutation(
                np.asarray(
                    list(itertools.combinations(set_parents,2))
                    )
                )
    n_pairs = np.size(pairs,axis=0)
    # creating arrays of pairs 
    PAIRS = np.zeros((n_pairs, n_elems, 2))
    for j in range(n_pairs):
        PAIRS[j,:,0] = H[:,pairs[j,0]]
        PAIRS[j,:,1] = H[:,pairs[j,1]]
    
    # making place for offspring
    H = H[:,:n_left]
    
    # the crossover (33/33/33 randomly chosen one-point, blend or two-point crossover)
    for j in range(n_pairs):
        
        if random.random() <= 1/3:
            C = OPCX(PAIRS[j,:,0], PAIRS[j,:,1])
            # opc = opc + 1
        elif random.random() <= 2/3:
            C = BLXa(PAIRS[j,:,0], PAIRS[j,:,1], h_min, h_max, 0.5) # why 0.67?
            # blx = blx + 1
        else:
            C = TPCX(PAIRS[j,:,0], PAIRS[j,:,1])
            # tpc += 1
        
        # depending on the number of left offspring, there is a little tournament between siblings or not
        if n_off_from_cross ==1:
            if obj_func(numfreq,C[:,0]) > obj_func(numfreq,C[:,1]):
                winner = C[:,0]
            else:
                winner = C[:,1]
            H = np.hstack((H,np.reshape(winner,(n_elems,-1))))
        else:
            H = np.hstack((H,C))

    H = H[:,:popul_size]

    return H, OF

