import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, days):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.arange(days)
    for _ in t[:-1]:
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dRdt = gamma * I[-1]
        S.append(S[-1] + dSdt)
        I.append(I[-1] + dIdt)
        R.append(R[-1] + dRdt)
    return S, I, R


S0 = 999
I0 = 1
R0 = 0
beta = 0.5
gamma = 0.1
days = 100

S, I, R = SIR_model(S0, I0, R0, beta, gamma, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
