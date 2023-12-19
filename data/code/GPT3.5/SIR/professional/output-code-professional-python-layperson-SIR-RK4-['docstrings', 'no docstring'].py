import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, t_end, dt):
    """
    Simulate SIR model using Runge-Kutta 4th order method.
    
    Parameters:
    beta : float
        Transmission rate
    gamma : float
        Recovery rate
    N : int
        Total population
    I0 : int
        Initial number of infected individuals
    R0 : int
        Initial number of recovered individuals
    t_end : float
        End time of simulation
    dt : float
        Time step size
    
    Returns:
    t : ndarray
        Array of time points
    S : ndarray
        Array of susceptible individuals
    I : ndarray
        Array of infected individuals
    R : ndarray
        Array of recovered individuals
    """
    
    def dSdt(S, I):
        return -beta * S * I / N
    
    def dIdt(S, I):
        return beta * S * I / N - gamma * I
    
    def dRdt(I):
        return gamma * I
    
    def rk4_step(S, I, R, dt):
        k1 = dt * dSdt(S, I)
        l1 = dt * dIdt(S, I)
        m1 = dt * dRdt(I)
        k2 = dt * dSdt(S + k1/2, I + l1/2)
        l2 = dt * dIdt(S + k1/2, I + l1/2)
        m2 = dt * dRdt(I + m1/2)
        k3 = dt * dSdt(S + k2/2, I + l2/2)
        l3 = dt * dIdt(S + k2/2, I + l2/2)
        m3 = dt * dRdt(I + m2/2)
        k4 = dt * dSdt(S + k3, I + l3)
        l4 = dt * dIdt(S + k3, I + l3)
        m4 = dt * dRdt(I + m3)
        S_next = S + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        I_next = I + (l1 + 2 * l2 + 2 * l3 + l4) / 6
        R_next = R + (m1 + 2 * m2 + 2 * m3 + m4) / 6
        return S_next, I_next, R_next
    
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        S[i], I[i], R[i] = rk4_step(S[i-1], I[i-1], R[i-1], dt)
    
    return t, S, I, R

