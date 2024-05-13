import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, I0, R0, duration):
    S0 = N - I0 - R0
    S = [S0]
    E = [0]
    I = [I0]
    R = [R0]
    t = np.arange(duration)

    for i in range(1, duration):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dIdt = sigma * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        S.append(S[i-1] + dSdt)
        E.append(E[i-1] + dEdt)
        I.append(I[i-1] + dIdt)
        R.append(R[i-1] + dRdt)

    return S, E, I, R


N = 1000
beta = 0.2
sigma = 1/5
gamma = 1/10
I0 = 50
R0 = 0
duration = 200

S, E, I, R = seir_model(N, beta, gamma, sigma, I0, R0, duration)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
