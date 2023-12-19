import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, S0, I0, R0, t_max):
    dt = 0.01
    t = np.arange(0, t_max+dt, dt)
    N = S0 + I0 + R0
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(len(t)-1):
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]
        S[i+1] = S[i] + dt * dSdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt
    return S, I, R


beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100

S, I, R = sir_model(beta, gamma, S0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
