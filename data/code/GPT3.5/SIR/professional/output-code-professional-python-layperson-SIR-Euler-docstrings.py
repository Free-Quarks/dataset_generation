import numpy as np
import matplotlib.pyplot as plt


def simulate_SIR_model(beta, gamma, S0, I0, R0, t_end, dt):
    '''
    Simulate and plot the SIR model using Euler's method
    
    Parameters:
    beta: float
        The average number of contacts per person per unit time multiplied by the probability of disease transmission
    gamma: float
        The recovery rate of infected individuals
    S0: int
        The initial number of susceptible individuals
    I0: int
        The initial number of infected individuals
    R0: int
        The initial number of recovered individuals
    t_end: float
        The end time of the simulation
    dt: float
        The time step size
    '''
    
    N = S0 + I0 + R0
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = (beta * S[i-1] * I[i-1] / N) - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        S[i] = S[i-1] + dS * dt
        I[i] = I[i-1] + dI * dt
        R[i] = R[i-1] + dR * dt
        
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()
}

