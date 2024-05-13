import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, R0, T):
    S0 = N - I0 - R0
    E0 = 0
    R = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    S = np.zeros(T)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    dt = 1

    for t in range(1, T):
        dSdt = -beta * S[t-1] * I[t-1] / N
        dEdt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dIdt = sigma * E[t-1] - gamma * I[t-1]
        dRdt = gamma * I[t-1]
        S[t] = S[t-1] + dt * dSdt
        E[t] = E[t-1] + dt * dEdt
        I[t] = I[t-1] + dt * dIdt
        R[t] = R[t-1] + dt * dRdt

    return S, E, I, R


beta = 0.2
sigma = 1/5
gamma = 1/10
N = 1000
I0 = 1
R0 = 0
T = 100

S, E, I, R = seir_model(beta, sigma, gamma, N, I0, R0, T)

plt.plot(np.arange(T), S, label='Susceptible')
plt.plot(np.arange(T), E, label='Exposed')
plt.plot(np.arange(T), I, label='Infected')
plt.plot(np.arange(T), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
