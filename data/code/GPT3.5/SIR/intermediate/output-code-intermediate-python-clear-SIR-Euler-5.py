import numpy as np
import matplotlib.pyplot as plt


def sir_euler(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    return S, I, R


# Example usage
S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = sir_euler(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
