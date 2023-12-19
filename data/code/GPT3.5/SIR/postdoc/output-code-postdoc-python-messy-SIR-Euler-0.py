import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, t_end, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    
    return S, I, R


# Example usage
S0 = 990
I0 = 10
R0 = 0
beta = 0.2
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
