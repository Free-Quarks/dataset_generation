import numpy as np
import matplotlib.pyplot as plt

def sir_model(S, I, R, beta, gamma):
    """
    SIR model dynamics
    
    Parameters:
    ----------
    S : float
        Number of susceptible individuals
    I : float
        Number of infected individuals
    R : float
        Number of recovered individuals
    beta : float
        Infection rate
    gamma : float
        Recovery rate
    
    Returns: 
    tuple
        Changes in the numbers of susceptible, infected and recovered individuals
    """
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return dS, dI, dR

def rk2(S, I, R, beta, gamma, dt):
    """
    Second-order Runge-Kutta method to solve SIR model
    
    Parameters:
    ----------
    S : float
        Initial number of susceptible individuals
    I : float
        Initial number of infected individuals
    R : float
        Initial number of recovered individuals
    beta : float
        Infection rate
    gamma : float
        Recovery rate
    dt : float
        Time step size
    
    Returns: 
    tuple
        Updated (S, I, R) values after one time step
    """
    dS1, dI1, dR1 = sir_model(S, I, R, beta, gamma)
    S1 = S + 0.5 * dt * dS1
    I1 = I + 0.5 * dt * dI1
    R1 = R + 0.5 * dt * dR1
    
    dS2, dI2, dR2 = sir_model(S1, I1, R1, beta, gamma)
    S += dt * dS2
    I += dt * dI2
    R += dt * dR2
    
    return S, I, R
