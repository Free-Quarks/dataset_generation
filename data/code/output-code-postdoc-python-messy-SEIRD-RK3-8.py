import numpy as np
import matplotlib.pyplot as plt


def seird_rk3(N, beta, gamma, delta, alpha, I0, E0, R0, D0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    D = np.zeros(len(t))

    # Set initial conditions
    S[0] = N - I0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0

    # Runge-Kutta 3rd order method
    for i in range(len(t) - 1):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - delta * E[i]
        dIdt = delta * E[i] - (1 - alpha) * gamma * I[i] - alpha * I[i]
        dRdt = (1 - alpha) * gamma * I[i]
        dDdt = alpha * I[i]

        S[i + 1] = S[i] + dt * dSdt
        E[i + 1] = E[i] + dt * dEdt
        I[i + 1] = I[i] + dt * dIdt
        R[i + 1] = R[i] + dt * dRdt
        D[i + 1] = D[i] + dt * dDdt

    return t, S, E, I, R, D


# Example usage
N = 100000
beta = 0.4
gamma = 0.1
delta = 0.2
alpha = 0.01
I0 = 100
E0 = 10
R0 = 0
D0 = 0

t_end = 100
dt = 0.1

t, S, E, I, R, D = seird_rk3(N, beta, gamma, delta, alpha, I0, E0, R0, D0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
