import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, S0, E0, I0, R0, t_max):
    N = S0 + E0 + I0 + R0
    dt = 0.1
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        E[i] = E[i-1] + dt * dE
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    return S, E, I, R


beta = 0.5
sigma = 0.1
gamma = 0.2
S0 = 900
E0 = 100
I0 = 10
R0 = 0
t_max = 100

S, E, I, R = seir_model(beta, gamma, sigma, S0, E0, I0, R0, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

