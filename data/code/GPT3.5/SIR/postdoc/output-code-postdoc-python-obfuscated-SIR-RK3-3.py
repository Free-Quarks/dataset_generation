import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, T):
    S0 = N - I0 - R0
    dt = 0.01
    steps = int(T / dt)
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    t = np.linspace(0, T, steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, steps):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    return S, I, R


beta = 0.3
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
T = 100

S, I, R = sir_model(beta, gamma, N, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
