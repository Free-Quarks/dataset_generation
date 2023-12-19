import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, t):
    S0 = N - I0 - R0
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = t[1] - t[0]

    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    return S, I, R


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t = np.linspace(0, 100, 1000)

S, I, R = sir_model(beta, gamma, N, I0, R0, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
