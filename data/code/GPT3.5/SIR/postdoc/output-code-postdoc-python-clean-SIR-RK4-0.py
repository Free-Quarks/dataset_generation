import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(S0, I0, R0, beta, gamma, t_end, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_end+1)
    I = np.zeros(t_end+1)
    R = np.zeros(t_end+1)
    t = np.linspace(0, t_end, t_end+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(t_end):
        dSdt = -beta/N * S[i] * I[i]
        dIdt = beta/N * S[i] * I[i] - gamma * I[i]
        dRdt = gamma * I[i]
        S[i+1] = S[i] + dt * dSdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt

    return S, I, R

# Example usage
S0 = 990
I0 = 10
R0 = 0
beta = 0.4
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_RK4(S0, I0, R0, beta, gamma, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using RK4')
plt.show()
