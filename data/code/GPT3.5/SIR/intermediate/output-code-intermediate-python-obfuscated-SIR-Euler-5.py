import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, population, days):
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = population - 1
    I[0] = 1
    R[0] = 0
    
    for t in range(days-1):
        S[t+1] = S[t] - beta * S[t] * I[t] / population
        I[t+1] = I[t] + beta * S[t] * I[t] / population - gamma * I[t]
        R[t+1] = R[t] + gamma * I[t]
    
    return S, I, R

beta = 0.3
gamma = 0.1
population = 1000
days = 100

S, I, R = SIR_model(beta, gamma, population, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
