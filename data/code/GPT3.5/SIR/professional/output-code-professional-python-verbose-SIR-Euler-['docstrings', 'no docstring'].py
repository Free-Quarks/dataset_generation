import numpy as np
import matplotlib.pyplot as plt

def simulate_sir_model(S0, I0, R0, beta, gamma, days):
    '''
    Simulate and plot the SIR model using Euler method
    
    Args:
        S0 (float): initial number of susceptible individuals
        I0 (float): initial number of infected individuals
        R0 (float): initial number of recovered individuals
        beta (float): transmission rate
        gamma (float): recovery rate
        days (int): number of days to simulate
    '''
    
    # Initialize arrays to store values
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    
    # Set initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Simulate the model
    for t in range(1, days):
        dS = -beta * S[t-1] * I[t-1]
        dI = beta * S[t-1] * I[t-1] - gamma * I[t-1]
        dR = gamma * I[t-1]
        
        S[t] = S[t-1] + dS
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR
    
    # Plot the results
    plt.plot(range(days), S, label='Susceptible')
    plt.plot(range(days), I, label='Infected')
    plt.plot(range(days), R, label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}

