import numpy as np
import matplotlib.pyplot as plt

def SIR_Euler(beta, gamma, N, I0, R0, days):
    S0 = N - I0 - R0
    t = np.linspace(0, days, days+1)
    S = np.zeros(days+1)
    I = np.zeros(days+1)
    R = np.zeros(days+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = 1

    for i in range(days):
        dS = -beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]
        S[i+1] = S[i] + dt * dS
        I[i+1] = I[i] + dt * dI
        R[i+1] = R[i] + dt * dR

    return S, I, R

# Example usage
beta = 0.5
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
days = 100

S, I, R = SIR_Euler(beta, gamma, N, I0, R0, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using Euler Method')
plt.show()

