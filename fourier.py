import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

#Compute and plot a f 2-step function starting at 0 on few period
def step_function(dx,period,upperval,lowerval,num_periods):
    num_samples_1p = int(period/dx)
    x=np.linspace(0, num_periods*period, num_samples_1p*num_periods)
    f=np.zeros_like(x)
    a = 0 
    inc = int(np.floor(num_samples_1p/2))
    b = a + inc
    for i in range(num_periods):
        f[a:b]=upperval
        f[b+1:b+1+inc]=lowerval
        a = a+2*inc
        b = b+2*inc
    return x,f

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

def fourier_coeff(f,x,P,num_harmonics):
    #We also store coefficient A0, B0
    size = num_harmonics+1 
    dx = 0.01
    num_samples_1p = int(P/dx)
    A = np.zeros([size])
    B = np.zeros([size])
    freqs = np.zeros([size])

    A[0] = 2/P*np.sum(f[:num_samples_1p]*np.ones_like(x[:num_samples_1p]))*dx
    B[0] = 0 
    freqs[0] = 0
    for i in range(1,size):
        #print('Compute '+ str(i)+'-th harmonic coefficient')
        freqs[i] = (2*i*np.pi/P)
        A[i] = (2/P)*np.sum(f[:num_samples_1p] * np.cos(freqs[i]*x[:num_samples_1p] ))*dx
        B[i] = (2/P)*np.sum(f[:num_samples_1p] * np.sin(freqs[i]*x[:num_samples_1p] ))*dx
    return A, B, freqs

def fourier_series(f,x,P,num_harmonics):
    A, B, freqs= fourier_coeff(f, x, P, num_harmonics)
    #H0 stores constant function
    size = num_harmonics+1 
    H = np.zeros([size,len(x)])
    SUM = np.zeros([len(x)])
    H[0] = A[0]/2
    SUM = H[0]
    for i in range(1,size):
        H[i] = A[i] * np.cos(freqs[i]*x)  + B[i] * np.sin((freqs[i])*x)
        SUM += H[i]

    return A, B, freqs, SUM, H

#Utility function to plot fourier results
def plot_fourier(f,x,A,B,freqs,H,SUM,num_harmonics):
    
    plt.rcParams['figure.figsize'] = [32,8]
    plt.rcParams.update({'font.size': 8})
    
    fig, ax = plt.subplots(1,4)
    cmap = get_cmap('tab20')
    colors = cmap.colors
    ax[2].set_prop_cycle(color=colors)

    n = len(x)

    ax[0].plot(x, f)
    ax[0].set_title('Function to approximate')
    
    ax[1].plot(x, SUM)
    ax[1].set_title('Fourier series')
    
    size = num_harmonics+1
    for k in range(1,size):
        ax[2].plot(x, H[k])
    ax[2].set_title('Number of Harmonics = ' + str(num_harmonics))
    
    

    #Plot harmonics A coefficients ( frequencies)
    ax[3].plot( freqs,A,'.')
    ax[3].set_title('Fourier coefficients')
    plt.tight_layout()
    plt.show()
    