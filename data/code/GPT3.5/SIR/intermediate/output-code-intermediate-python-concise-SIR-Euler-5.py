import numpy as np
import matplotlib.pyplot as plt


def sir_model(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    
    for t in range(days):
        S_t = S[-1]
        I_t = I[-1]
        R_t = R[-1]
        
        dS = -beta * S_t * I_t / N
        dI = beta * S_t * I_t / N - gamma * I_t
        dR = gamma * I_t
        
        S.append(S_t + dS)
        I.append(I_t + dI)
        R.append(R_t + dR)
    
    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
days = 100

S, I, R = sir_model(N, I0, R0, beta, gamma, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
