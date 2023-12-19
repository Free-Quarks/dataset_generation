import numpy as np
import matplotlib.pyplot as plt

def simulate_SIR(beta, gamma, S0, I0, R0, num_days):
    N = S0 + I0 + R0
    S = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for t in range(1, num_days):
        dSdt = -beta * S[t-1] * I[t-1] / N
        dIdt = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dRdt = gamma * I[t-1]
        
        S[t] = S[t-1] + dSdt
        I[t] = I[t-1] + dIdt
        R[t] = R[t-1] + dRdt
    
    return S, I, R


# Example usage
beta = 0.3
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
num_days = 100

S, I, R = simulate_SIR(beta, gamma, S0, I0, R0, num_days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
