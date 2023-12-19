import numpy as np
import matplotlib.pyplot as plt


def SIR_model():
    N = 1000  # Total population
    I0 = 1  # Initial number of infected individuals
    R0 = 0  # Initial number of recovered individuals
    S0 = N - I0 - R0  # Initial number of susceptible individuals
    beta = 0.2  # Contact rate
    gamma = 0.1  # Recovery rate
    t_end = 100  # Number of time steps
    dt = 0.1  # Time step size

    t = np.linspace(0, t_end, int(t_end/dt)+1)  # Time vector
    S = np.zeros_like(t)  # Susceptible
    I = np.zeros_like(t)  # Infected
    R = np.zeros_like(t)  # Recovered

    S[0], I[0], R[0] = S0, I0, R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


SIR_model()
