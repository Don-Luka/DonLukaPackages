import numpy as np
import matplotlib.pyplot as plt

def symmetrize_y(h, n_x, n_y):
    h0 = h.reshape((n_y-1,-1))
    i0 = np.arange((n_x-1)*(n_y-1)).reshape((n_y-1,-1))
    i1 = [j for j in range(int(n_y/2))]
    i2 = [-j-1 for j in range(int((n_y-1)/2))]
    for j in range(int((n_y-1)/2)):
        h0[i2[j]]=h0[i1[j]]
    return h

if __name__ == '__main__':
    n_x, n_y = 101,11
    A = np.random.randint(low=2,high=4, size=(n_x-1)*(n_y-1))
    print(f'{A=}')
    A=symmetrize_y(A, n_x, n_y)\
        .reshape((n_y-1,-1))
    print(A)
    plt.imshow(A)
    plt.show()    