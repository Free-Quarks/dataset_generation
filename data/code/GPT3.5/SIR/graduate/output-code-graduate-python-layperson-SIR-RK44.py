import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 4th order method
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    # Create plot
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

SIR_model(beta, gamma, N, I0, R0, t_end, dt)
