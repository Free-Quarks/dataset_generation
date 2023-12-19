import numpy as np
import matplotlib.pyplot as plt


def sir_model(N, I0, R0, beta, gamma, t): 
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]

    dt = t[1] - t[0]
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        S.append(S[i-1] + dt * dS)
        I.append(I[i-1] + dt * dI)
        R.append(R[i-1] + dt * dR)

    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, 100)

S, I, R = sir_model(N, I0, R0, beta, gamma, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
