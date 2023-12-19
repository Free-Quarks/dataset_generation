import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, timesteps):
    """
    Simulate and plot the SIR model.

    Parameters:
    beta (float): The infection rate
    gamma (float): The recovery rate
    S0 (int): Initial number of susceptible individuals
    I0 (int): Initial number of infected individuals
    R0 (int): Initial number of recovered individuals
    timesteps (int): Number of time steps for the simulation
    """
    N = S0 + I0 + R0
    S = np.zeros(timesteps)
    I = np.zeros(timesteps)
    R = np.zeros(timesteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = 1
    for t in range(1, timesteps):
        dS_dt = -beta * S[t-1] * I[t-1] / N
        dI_dt = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR_dt = gamma * I[t-1]
        S[t] = S[t-1] + dt * dS_dt
        I[t] = I[t-1] + dt * dI_dt
        R[t] = R[t-1] + dt * dR_dt

    plt.plot(range(timesteps), S, label='Susceptible')
    plt.plot(range(timesteps), I, label='Infected')
    plt.plot(range(timesteps), R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}

