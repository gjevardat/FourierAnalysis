import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def square_function():
    dx = 0.01
    x = np.arange(-1, 1, dx)
    f = np.zeros_like(x)
    n = len(x)
    nquart = int(np.floor(n / 4))

    f[:-1 + nquart] = 0
    f[-1 + nquart:1 - nquart] = 1
    f[1 - nquart:] = 0

    return x, dx, f

#Simple P-periodic triangle function defined on interval a,b and centered  on x0
def triangle_function(dx,P,x0):
    
    x = np.arange(x0-P/2, x0+P/2, dx)
    f = np.zeros_like(x)
    n = len(x)
    nquart = int(np.floor(n / 4))

    step = P/len(x)
    f[:nquart] = 0
    f[nquart:2 * nquart] = np.linspace(0, 1, num=nquart)
    f[2 * nquart:3 * nquart] = np.ones(nquart) - (np.linspace(0, 1, num=nquart))
    f[3 * nquart:] = 0
    return x, dx, f

#Compute the first k fourier coefficients of a P periodic function 
def compute_fourrier_coef(f,x,dx,P,k):
    A = np.zeros([k])
    B = np.zeros([k])
    freqs = np.zeros([k])
    H = np.zeros([k,len(x)])
    SUM = np.zeros([len(x)])
    
    A[0] = np.sum(f*np.ones_like(x)* dx)  / (P)
    print(A[0])
    B[0] = 0 
    for i in range(1,k):
        freqs[i] = (i*np.pi/P)
        A[i] = np.sum(f * np.cos(freqs[i]*x ))*dx/P
        B[i] = np.sum(f * np.sin(freqs[i]*x ))*dx/P
        H[i] = A[i] * np.cos((i*np.pi/P)*x)  + B[i] * np.sin((i*np.pi/P)*x)
        SUM += H[i]

    H[0] = A[0]/2
    SUM += H[0]
    
    return A, B, freqs, H, SUM

#Utility function to plot fourier results
def plot_fourier(f,x,A,B,freqs,H,SUM,num_harmonics):
    
    plt.rcParams['figure.figsize'] = [12,12]
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(2, 2)
    cmap = get_cmap('tab20')
    colors = cmap.colors
    ax[0][1].set_prop_cycle(color=colors)

    n = len(x)

    ax[0][0].plot(x, f)
    ax[0][0].set_title('Function to approximate')
    
    for k in range(1,num_harmonics):
        ax[0][1].plot(x, H[k])
        ax[0][1].set_title('Number of Harmonics = ' + str(num_harmonics))
    
    ax[1][0].plot(x, SUM)
    ax[1][0].set_title('Fourier series')

    #Plot harmonics A coefficients ( frequencies)
    ax[1][1].plot( freqs[1:],A[1:])
    ax[1][1].set_title('Fourier coefficients')
    plt.tight_layout()
    plt.show()
    