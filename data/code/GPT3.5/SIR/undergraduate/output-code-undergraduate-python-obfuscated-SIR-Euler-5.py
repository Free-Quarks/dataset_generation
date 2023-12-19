import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.arange(days)
    
    for _ in range(days-1):
        S_next = S[-1] - (beta * S[-1] * I[-1] / N)
        I_next = I[-1] + (beta * S[-1] * I[-1] / N) - (gamma * I[-1])
        R_next = R[-1] + (gamma * I[-1])
        
        S.append(S_next)
        I.append(I_next)
        R.append(R_next)
    
    return S, I, R

N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
R0 = 0
days = 100

S, I, R = SIR_model(beta, gamma, N, I0, R0, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
