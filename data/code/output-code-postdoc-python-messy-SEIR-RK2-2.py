import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, R0, T):
    S0 = N - I0 - R0
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    S = np.zeros(T+1)
    E = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    S[0] = S0
    E[0] = 0
    I[0] = I0
    R[0] = R0
    for i in range(T):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]
        S[i+1] = S[i] + dt * dSdt
        E[i+1] = E[i] + dt * dEdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt
    return t, S, E, I, R

beta = 0.2
sigma = 0.1
gamma = 0.05
N = 1000
I0 = 1
R0 = 0
T = 100
t, S, E, I, R = seir_model(beta, sigma, gamma, N, I0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
