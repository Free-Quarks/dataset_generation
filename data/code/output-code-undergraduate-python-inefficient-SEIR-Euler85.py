import numpy as np
import matplotlib.pyplot as plt

def SEIR_model(N, beta, gamma, sigma, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    E = [0]
    I = [I0]
    R = [R0]
    
    for day in range(days):
        S_to_E = beta * S[-1] * I[-1] / N
        E_to_I = sigma * E[-1]
        I_to_R = gamma * I[-1]
        
        S_new = S[-1] - S_to_E
        E_new = E[-1] + S_to_E - E_to_I
        I_new = I[-1] + E_to_I - I_to_R
        R_new = R[-1] + I_to_R
        
        S.append(S_new)
        E.append(E_new)
        I.append(I_new)
        R.append(R_new)
        
    return S, E, I, R


# example usage
N = 10000
beta = 0.25
gamma = 0.1
sigma = 0.2
I0 = 10
R0 = 0
days = 100

S, E, I, R = SEIR_model(N, beta, gamma, sigma, I0, R0, days)

plt.plot(range(days + 1), S, label='Susceptible')
plt.plot(range(days + 1), E, label='Exposed')
plt.plot(range(days + 1), I, label='Infected')
plt.plot(range(days + 1), R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
