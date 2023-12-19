import numpy as np


def SIR_RK4(N, I0, R0, beta, gamma, t_max, dt):
    '''
    Simulate the SIR model using 4th order Runge-Kutta method.
    
    Parameters:
    N: int
        Total population size
    I0: int
        Initial number of infected individuals
    R0: int
        Initial number of recovered individuals
    beta: float
        Transmission rate
    gamma: float
        Recovery rate
    t_max: float
        Maximum time of simulation
    dt: float
        Time step size
    
    Returns:
    S: array
        Array containing the number of susceptible individuals at each time step
    I: array
        Array containing the number of infected individuals at each time step
    R: array
        Array containing the number of recovered individuals at each time step
    '''
    
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    # Define the differential equations
    def dSdt(S, I):
        return -beta * (S * I / N)
    
    def dIdt(S, I):
        return beta * (S * I / N) - gamma * I
    
    def dRdt(I):
        return gamma * I
    
    # Implement the 4th order Runge-Kutta method
    for i in range(1, len(t)):
        k1_S = dt * dSdt(S[i-1], I[i-1])
        k1_I = dt * dIdt(S[i-1], I[i-1])
        k1_R = dt * dRdt(I[i-1])
        
        k2_S = dt * dSdt(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I)
        k2_I = dt * dIdt(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I)
        k2_R = dt * dRdt(I[i-1] + 0.5 * k1_I)
        
        k3_S = dt * dSdt(S[i-1] + 0.5 * k2_S, I[i-1] + 0.5 * k2_I)
        k3_I = dt * dIdt(S[i-1] + 0.5 * k2_S, I[i-1] + 0.5 * k2_I)
        k3_R = dt * dRdt(I[i-1] + 0.5 * k2_I)
        
        k4_S = dt * dSdt(S[i-1] + k3_S, I[i-1] + k3_I)
        k4_I = dt * dIdt(S[i-1] + k3_S, I[i-1] + k3_I)
        k4_R = dt * dRdt(I[i-1] + k3_I)
        
        S[i] = S[i-1] + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
        I[i] = I[i-1] + (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
        R[i] = R[i-1] + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6
    
    return S, I, R

