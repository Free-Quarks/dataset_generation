import numpy as np
import matplotlib.pyplot as plt

def SIR_model(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 1
    t = np.linspace(0, days, int(days/dt) + 1)

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S.append(S[i-1] + dS * dt)
        I.append(I[i-1] + dI * dt)
        R.append(R[i-1] + dR * dt)

    return S, I, R

# Example usage:
N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
days = 100

S, I, R = SIR_model(N, I0, R0, beta, gamma, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
