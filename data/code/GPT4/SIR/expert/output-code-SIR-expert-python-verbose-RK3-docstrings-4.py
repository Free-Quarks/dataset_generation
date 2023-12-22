import numpy as np
import json

def runge_kutta3_SIR(susceptible, infected, recovered, beta, gamma, dt):
    """
    Runge-Kutta 3rd order method to solve SIR model
    Args:
    susceptible : initial susceptible population
    infected : initial infected population
    recovered : initial recovered population
    beta : infection rate
    gamma : recovery rate
    dt : time step

    Returns:
    updated populations after one time step
    """
    
    # Runge-Kutta 3rd order method applied to each equation in the SIR model
    k1_S = -beta*susceptible*infected
    k1_I = beta*susceptible*infected - gamma*infected
    k1_R = gamma*infected
    
    k2_S = -beta*(susceptible+k1_S/2)*(infected+k1_I/2)
    k2_I = beta*(susceptible+k1_S/2)*(infected+k1_I/2) - gamma*(infected+k1_I/2)
    k2_R = gamma*(infected+k1_I/2)
    
    k3_S = -beta*(susceptible-k1_S+2*k2_S)*(infected-k1_I+2*k2_I)
    k3_I = beta*(susceptible-k1_S+2*k2_S)*(infected-k1_I+2*k2_I) - gamma*(infected-k1_I+2*k2_I)
    k3_R = gamma*(infected-k1_I+2*k2_I)
    
    next_susceptible = susceptible + dt/6*(k1_S+4*k2_S+k3_S)
    next_infected = infected + dt/6*(k1_I+4*k2_I+k3_I)
    next_recovered = recovered + dt/6*(k1_R+4*k2_R+k3_R)

    return next_susceptible, next_infected, next_recovered
