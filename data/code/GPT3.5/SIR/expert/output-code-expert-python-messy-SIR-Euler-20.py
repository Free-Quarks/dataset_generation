import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, T, I0, N):
    dt = 0.01
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    return t, S, I, R


beta = 0.2
gamma = 0.1
T = 100
I0 = 1
N = 1000

t, S, I, R = sir_model(beta, gamma, T, I0, N)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

