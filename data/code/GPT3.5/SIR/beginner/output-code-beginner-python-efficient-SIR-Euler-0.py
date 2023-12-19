import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, t_end, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_end)  # susceptible
    I = np.zeros(t_end)  # infected
    R = np.zeros(t_end)  # recovered
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for t in range(1, t_end):
        dSdt = -beta * S[t-1] * I[t-1] / N
        dIdt = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dRdt = gamma * I[t-1]
        
        S[t] = S[t-1] + dt * dSdt
        I[t] = I[t-1] + dt * dIdt
        R[t] = R[t-1] + dt * dRdt
    
    return S, I, R


# Example usage
S0 = 1000  # Initial number of susceptibles
I0 = 1  # Initial number of infected
R0 = 0  # Initial number of recovered
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
t_end = 100  # Time period
dt = 1  # Time step

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
