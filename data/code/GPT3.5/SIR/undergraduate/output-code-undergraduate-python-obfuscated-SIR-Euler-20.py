import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 1

    for day in range(days-1):
        dS = -beta * S[day] * I[day] / N
        dI = (beta * S[day] * I[day] / N) - gamma * I[day]
        dR = gamma * I[day]
        S.append(S[day] + dt * dS)
        I.append(I[day] + dt * dI)
        R.append(R[day] + dt * dR)

    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
days = 100

S, I, R = SIR_model(beta, gamma, N, I0, R0, days)

plt.plot(range(days), S, label='Susceptible')
plt.plot(range(days), I, label='Infected')
plt.plot(range(days), R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
