import numpy as np
import matplotlib.pyplot as plt

def SIDARTHE_RK2(initial_conditions, parameters, t_max, dt):
    # Unpack initial conditions
    S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0 = initial_conditions
    
    # Unpack parameters
    beta, gamma, delta, alpha, rho, theta, eta, mu, nu, k, p = parameters
    
    # Define the system of ODEs
    def dS_dt(S, I, D, A, R, T, H, E, t):
        N = S + I + D + A + R + T + H + E
        dS = -beta*S*(I + alpha*A)/N
        return dS

    def dI_dt(S, I, D, A, R, T, H, E, t):
        N = S + I + D + A + R + T + H + E
        dI = beta*S*(I + alpha*A)/N - (gamma + delta)*I
        return dI

    def dD_dt(S, I, D, A, R, T, H, E, t):
        dD = delta*I - (rho + theta + eta)*D
        return dD

    def dA_dt(S, I, D, A, R, T, H, E, t):
        dA = alpha*beta*S*(I + alpha*A)/N - (mu + nu)*A
        return dA

    def dR_dt(S, I, D, A, R, T, H, E, t):
        dR = gamma*I + rho*D - (k + p)*R
        return dR

    def dT_dt(S, I, D, A, R, T, H, E, t):
        dT = theta*D - T
        return dT

    def dH_dt(S, I, D, A, R, T, H, E, t):
        dH = eta*D - H
        return dH

    def dE_dt(S, I, D, A, R, T, H, E, t):
        dE = mu*A - E
        return dE

    # Perform integration using RK2 method
    num_steps = int(t_max/dt)
    t = np.linspace(0, t_max, num_steps+1)
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    D = np.zeros(num_steps+1)
    A = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)
    T = np.zeros(num_steps+1)
    H = np.zeros(num_steps+1)
    E = np.zeros(num_steps+1)
    
    S[0], I[0], D[0], A[0], R[0], T[0], H[0], E[0] = initial_conditions
    
    for i in range(num_steps):
        t_i = t[i]
        S_i = S[i]
        I_i = I[i]
        D_i = D[i]
        A_i = A[i]
        R_i = R[i]
        T_i = T[i]
        H_i = H[i]
        E_i = E[i]
        
        # First RK2 step
        S_half = S_i + 0.5*dt*dS_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        I_half = I_i + 0.5*dt*dI_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        D_half = D_i + 0.5*dt*dD_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        A_half = A_i + 0.5*dt*dA_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        R_half = R_i + 0.5*dt*dR_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        T_half = T_i + 0.5*dt*dT_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        H_half = H_i + 0.5*dt*dH_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        E_half = E_i + 0.5*dt*dE_dt(S_i, I_i, D_i, A_i, R_i, T_i, H_i, E_i, t_i)
        
        # Second RK2 step
        S[i+1] = S_i + dt*dS_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        I[i+1] = I_i + dt*dI_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        D[i+1] = D_i + dt*dD_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        A[i+1] = A_i + dt*dA_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        R[i+1] = R_i + dt*dR_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        T[i+1] = T_i + dt*dT_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        H[i+1] = H_i + dt*dH_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        E[i+1] = E_i + dt*dE_dt(S_half, I_half, D_half, A_half, R_half, T_half, H_half, E_half, t_i + 0.5*dt)
        
    # Prepare results dictionary
    results = {
        't': t,
        'S': S,
        'I': I,
        'D': D,
        'A': A,
        'R': R,
        'T': T,
        'H': H,
        'E': E
    }
    
    return results

