import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, N, dt, t_max):
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0], I[0], R[0] = S0, I0, R0

    for i in range(1, len(t)):
        delta_S = -beta * S[i-1] * I[i-1] / N
        delta_I = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        delta_R = gamma * I[i-1]

        S[i] = S[i-1] + (delta_S * dt)
        I[i] = I[i-1] + (delta_I * dt)
        R[i] = R[i-1] + (delta_R * dt)

    return S, I, R


S0, I0, R0 = 999, 1, 0  # initial conditions
beta, gamma = 0.3, 0.1  # infection and recovery rates
N = S0 + I0 + R0  # total population

# Simulation parameters
dt = 0.1  # time step
t_max = 100  # maximum time

def plot_SIR(S, I, R, t):
    plt.figure(figsize=(12, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()


S, I, R = SIR_model(S0, I0, R0, beta, gamma, N, dt, t_max)
plot_SIR(S, I, R, t)
