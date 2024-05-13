import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, R0, timesteps):
    S0 = N - I0 - R0
    S = [S0]
    E = [0]
    I = [I0]
    R = [R0]
    dt = 1
    for t in range(timesteps):
        dS = -beta * S[-1] * I[-1] / N
        dE = beta * S[-1] * I[-1] / N - sigma * E[-1]
        dI = sigma * E[-1] - gamma * I[-1]
        dR = gamma * I[-1]
        S.append(S[-1] + dt * dS)
        E.append(E[-1] + dt * dE)
        I.append(I[-1] + dt * dI)
        R.append(R[-1] + dt * dR)
    return S, E, I, R

N = 1000
I0 = 1
R0 = 0
beta = 0.2
sigma = 0.1
gamma = 0.05
timesteps = 100

S, E, I, R = seir_model(beta, sigma, gamma, N, I0, R0, timesteps)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

