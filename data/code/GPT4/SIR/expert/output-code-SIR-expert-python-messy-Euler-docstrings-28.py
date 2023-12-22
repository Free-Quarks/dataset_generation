import numpy as np
import matplotlib.pyplot as plt

def euler_SIR(susceptible, infected, recovered, beta, gamma, time):
    """ 
    A function to simulate an SIR model using Euler's method.
    
    Parameters:
    susceptible (int): Initial number of susceptible individuals
    infected (int): Initial number of infected individuals
    recovered (int): Initial number of recovered individuals
    beta (float): Parameter representing the effective contact rate
    gamma (float): Parameter representing the recovery rate
    time (int): Number of time steps to simulate
    
    Returns:
    A plot showing the number of susceptible, infected, and recovered individuals over time
    """
    
    S = np.zeros(time)
    I = np.zeros(time)
    R = np.zeros(time)
    
    S[0] = susceptible
    I[0] = infected
    R[0] = recovered
    
    for t in range(time-1):
        S[t+1] = S[t] - beta*S[t]*I[t]
        I[t+1] = I[t] + beta*S[t]*I[t] - gamma*I[t]
        R[t+1] = R[t] + gamma*I[t]
    
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.legend()
    plt.show()
