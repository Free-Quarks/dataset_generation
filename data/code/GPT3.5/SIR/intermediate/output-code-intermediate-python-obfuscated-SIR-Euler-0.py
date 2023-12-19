import numpy as np
import matplotlib.pyplot as plt


def SIR_Euler(beta, gamma, S0, I0, R0, t_end, dt):
    t = np.arange(0, t_end, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
    
    return S, I, R


beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_end = 100
dt = 1

S, I, R = SIR_Euler(beta, gamma, S0, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using Euler Method')
plt.legend()
plt.show()
