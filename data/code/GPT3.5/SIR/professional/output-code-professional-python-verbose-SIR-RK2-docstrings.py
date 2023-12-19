import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, T):
    """
    Simulate and plot the SIR compartmental model.

    Parameters:
    - beta: float, the transmission rate
    - gamma: float, the recovery rate
    - S0: int, the initial number of susceptible individuals
    - I0: int, the initial number of infected individuals
    - R0: int, the initial number of recovered individuals
    - T: int, the total number of time steps

    Returns:
    - S: numpy array, the number of susceptible individuals at each time step
    - I: numpy array, the number of infected individuals at each time step
    - R: numpy array, the number of recovered individuals at each time step
    """
    dt = 0.1
    N = S0 + I0 + R0
    S = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for t in range(1, T):
        dS = -beta * S[t - 1] * I[t - 1] / N
        dI = beta * S[t - 1] * I[t - 1] / N - gamma * I[t - 1]
        dR = gamma * I[t - 1]
        
        S[t] = S[t - 1] + dt * dS
        I[t] = I[t - 1] + dt * dI
        R[t] = R[t - 1] + dt * dR

    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
T = 100

# Run the simulation
S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
