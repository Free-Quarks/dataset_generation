import numpy as np


def SIDARTHE_RK4(N, beta, gamma, delta, theta, epsilon, rho, sigma, kappa, xi, mu, tmax, dt):
    
    def rhs(t, y):
        S = y[0]
        I = y[1]
        D = y[2]
        A = y[3]
        R = y[4]
        T = y[5]
        H = y[6]
        E = y[7]
        Ia = y[8]
        Ih = y[9]
        
        dSdt = -beta * S * (I + delta * A + theta * H) / N
        dIdt = beta * S * (I + delta * A + theta * H) / N - (gamma + epsilon + rho) * I
        dDdt = rho * I
        dAdt = epsilon * I - (sigma + kappa) * A
        dRdt = gamma * I + sigma * A
        dTdt = delta * A + theta * H
        dHdt = kappa * A
        dEdt = mu * (Ia + Ih) - xi * E
        dIadt = xi * E - mu * Ia
        dIhdt = xi * E - mu * Ih
        
        return np.array([dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt, dIadt, dIhdt])
    
    y0 = np.array([N-1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    t = np.arange(0, tmax, dt)
    n = len(t)
    y = np.zeros((n, 10))
    
    y[0] = y0
    
    for i in range(n-1):
        k1 = dt * rhs(t[i], y[i])
        k2 = dt * rhs(t[i] + 0.5*dt, y[i] + 0.5*k1)
        k3 = dt * rhs(t[i] + 0.5*dt, y[i] + 0.5*k2)
        k4 = dt * rhs(t[i] + dt, y[i] + k3)
        
        y[i+1] = y[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    S = y[:,0]
    I = y[:,1]
    D = y[:,2]
    A = y[:,3]
    R = y[:,4]
    T = y[:,5]
    H = y[:,6]
    E = y[:,7]
    Ia = y[:,8]
    Ih = y[:,9]
    
    return S, I, D, A, R, T, H, E, Ia, Ih
