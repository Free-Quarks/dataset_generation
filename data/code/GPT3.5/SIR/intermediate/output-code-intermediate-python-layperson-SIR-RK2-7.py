import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, S0, I0, R0, N, t_max):
    dt = 0.01
    t = np.arange(0, t_max+dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
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
beta = 0.2
gamma = 0.1
N = 1000
S0 = N - 1
I0 = 1
R0 = 0
t_max = 100

S, I, R = sir_model(beta, gamma, S0, I0, R0, N, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
