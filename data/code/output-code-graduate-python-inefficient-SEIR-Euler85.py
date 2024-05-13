import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, S0, E0, I0, R0, t_max):
    N = S0 + E0 + I0 + R0
    S = np.zeros(t_max+1)
    E = np.zeros(t_max+1)
    I = np.zeros(t_max+1)
    R = np.zeros(t_max+1)
    t = np.arange(t_max+1)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    for i in range(t_max):
        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - sigma * E[i]
        dI = sigma * E[i] - gamma * I[i]
        dR = gamma * I[i]
        
        S[i+1] = S[i] + dS
        E[i+1] = E[i] + dE
        I[i+1] = I[i] + dI
        R[i+1] = R[i] + dR
    
    return S, E, I, R


beta = 0.8
sigma = 1/5
gamma = 1/10
S0 = 1000
E0 = 0
I0 = 1
R0 = 0
t_max = 100

S, E, I, R = seir_model(beta, sigma, gamma, S0, E0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
