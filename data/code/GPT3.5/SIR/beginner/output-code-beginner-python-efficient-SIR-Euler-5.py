import numpy as np
import matplotlib.pyplot as plt

def sir_model(N, beta, gamma, I0, T):
    S0 = N - I0
    R0 = 0
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = 1
    for i in range(T):
        dS = -beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]
        S[i+1] = S[i] + dt * dS
        I[i+1] = I[i] + dt * dI
        R[i+1] = R[i] + dt * dR
    return t, S, I, R

N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
T = 100

t, S, I, R = sir_model(N, beta, gamma, I0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
