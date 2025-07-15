import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.fft as fft
import scipy.interpolate as interpolate
import statsmodels.api as sm
from scipy.integrate import solve_ivp
import time # to obtain the running time of some functions


def generic_CF(u, params, S0, r, q, T, model):
    ''' Computes the characteristic function for different models (BMS, Heston, VG). '''   
    
    if (model == 'BMS'):
        # unpack parameters
        sig = params[0]
        # cf
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
    
    elif(model == 'Heston'):  
        # unpack parameters
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        # ChF
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)        
        pow1 = 2*kappa*theta/(sigma**2)
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*np.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)

    elif (model == 'VG'):
        # unpack parameters
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];
        # cf
        if (nu == 0):
            mu = np.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = np.exp(1j*u*mu) * np.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = np.log(S0) + (r-q + np.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = np.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))

    return phi


def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    ''' Option pricing using FFT (model-free). '''
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2 * np.pi / N) / eta
    
    # choice of beta
    #beta = np.log(S0)-N*lda/2 # the log strike we want is in the middle of the array
    beta = np.log(K) # the log strike we want is the first element of the array
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros(N)
    xX = np.zeros(N)
    
    # discount factor
    df = np.exp(-r*T)
    
    nuJ = np.arange(N) * eta
    psi_nuJ = generic_CF(nuJ - (alpha + 1) * 1j, params, S0, r, q, T, model) / ((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    km = beta + lda * np.arange(N)
    w = eta * np.ones(N)
    w[0] = eta / 2
    xX = np.exp(-1j * beta * nuJ) * df * psi_nuJ * w
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros(N) 
    multiplier = np.exp(-alpha * km) / np.pi
    cT_km = multiplier * np.real(yY)
    
    return km, cT_km



def odeRK4(fcn, a, b, y0, N, *var):
    """
    ODE solver with RK4 algorithm
    Solve dY(t)/dt = f(t, Y(t)) in 'N' steps using 4th-order Runge-Kutta witdt initial condition Y[a] = y0.
    where Y(t,u), given each u, is the desired coefficient function, e.g. A(t,u), B(t,u)
    
    Parameters:
    fcn: RHS function in coefficient functions
    a,b: time interval; usually set a=0(meaning tau=0), b=T; 以到期日条件作为初始条件
    N: # of discretisation steps

    Return:
    t, Y(t,u): time path; coef function path along t, with given u
    """
    dt = (b - a) / N
    t = a + np.arange(N + 1) * dt
    nDim = var[0] # dimention of u (in ChF)
    y = np.zeros([int(nDim), t.size], dtype=complex)
    y[:,0] = y0

    for k in range(N):
        k1 = fcn(t[k], y[:,k])
        k2 = fcn(t[k] + dt / 2, y[:,k] + dt * k1 / 2)
        k3 = fcn(t[k] + dt / 2, y[:,k] + dt * k2 / 2)
        k4 = fcn(t[k] + dt, y[:,k] + dt * k3)
        y[:,k+1] = y[:,k] + dt * (k1 + 2 * (k2 + k3) + k4) / 6

    return t, y
  
  
  
def GeneratePathsHestonEuler(NoOfPaths,NoOfSteps,T,r,S_0,kappa,gamma,rho,vbar,v0):
    '''
    Euler scheme for Heston1993 model 
    '''
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0
    X[:,0]=np.log(S_0)
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
        Z2[:,i] = rho * Z1[:,i] + np.sqrt(1.0-rho**2)*Z2[:,i]
        
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        
        # Truncated boundary condition
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W1[:,i+1]-W1[:,i])
        V[:,i+1] = np.maximum(V[:,i+1],0.0)
        
        X[:,i+1] = X[:,i] + (r - 0.5*V[:,i])*dt + np.sqrt(V[:,i])*(W2[:,i+1]-W2[:,i])
        time[i+1] = time[i] +dt
        
    #Compute exponent
    S = np.exp(X)
    # paths = {"time":time,"S":S}
    return time, S  
  
