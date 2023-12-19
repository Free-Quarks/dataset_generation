import numpy as np
import matplotlib.pyplot as plt

def SIR_model(N, beta, gamma, S0, I0, R0, days):
    S = [S0]
    I = [I0]
    R = [R0]

    for _ in range(days):
        dS = -beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR = gamma * I[-1]

        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)

    return S, I, R

N = 100000
beta = 0.3
gamma = 0.1
S0 = 99999
I0 = 1
R0 = 0
days = 100

S, I, R = SIR_model(N, beta, gamma, S0, I0, R0, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
