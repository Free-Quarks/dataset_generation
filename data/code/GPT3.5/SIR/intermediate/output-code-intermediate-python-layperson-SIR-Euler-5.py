import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, N, days):
    
    S = [S0]
    I = [I0]
    R = [R0]
    
    dt = 1
    
    for t in range(days):
        dS = -beta * S[t] * I[t] / N
        dI = beta * S[t] * I[t] / N - gamma * I[t]
        dR = gamma * I[t]
        
        S.append(S[t] + dt * dS)
        I.append(I[t] + dt * dI)
        R.append(R[t] + dt * dR)
        
    return S, I, R


# Example usage:

S0 = 999
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
N = S0 + I0 + R0
days = 100

S, I, R = SIR_model(S0, I0, R0, beta, gamma, N, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
