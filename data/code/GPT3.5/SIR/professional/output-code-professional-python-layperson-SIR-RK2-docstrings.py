import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    """
    Simulates the SIR model using the Runge-Kutta second order method.

    Parameters:
    - beta: float, the transmission rate
    - gamma: float, the recovery rate
    - N: int, the total population
    - I0: int, the initial number of infected individuals
    - R0: int, the initial number of recovered individuals
    - t_end: float, the end time of the simulation
    - dt: float, the time step size

    Returns:
    - S: array, the time evolution of the susceptible individuals
    - I: array, the time evolution of the infected individuals
    - R: array, the time evolution of the recovered individuals
    - t: array, the time points at which the populations are computed
    """
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta second order
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * (dSdt + (-beta * (S[i-1] + dt * dSdt / 2) * (I[i-1] + dt * dIdt / 2) / N)) / 2
        I[i] = I[i-1] + dt * (dIdt + (beta * (S[i-1] + dt * dSdt / 2) * (I[i-1] + dt * dIdt / 2) / N - gamma * (I[i-1] + dt * dIdt / 2))) / 2
        R[i] = R[i-1] + dt * (dRdt + (gamma * (I[i-1] + dt * dIdt / 2))) / 2

    return S, I, R, t


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R, t = SIR_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
