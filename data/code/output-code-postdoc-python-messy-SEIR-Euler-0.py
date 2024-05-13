import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    dt = T[1] - T[0]
    for t in range(1, len(T)):
        dS = -beta * S[t-1] * I[t-1] / N
        dE = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dI = sigma * E[t-1] - gamma * I[t-1]
        dR = gamma * I[t-1]
        S.append(S[t-1] + dt * dS)
        E.append(E[t-1] + dt * dE)
        I.append(I[t-1] + dt * dI)
        R.append(R[t-1] + dt * dR)
    return S, E, I, R


# Example usage
beta = 0.2
sigma = 0.1
gamma = 0.05
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = np.linspace(0, 100, 1000)

S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
