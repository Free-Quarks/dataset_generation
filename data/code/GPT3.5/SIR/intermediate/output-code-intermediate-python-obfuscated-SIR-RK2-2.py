import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, N, t_end, dt):
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2) / N
        k2_I = beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2) / N - gamma * (I[i-1] + dt * k1_I/2)
        
        S[i] = S[i-1] + dt * k2_S
        I[i] = I[i-1] + dt * k2_I
        R[i] = R[i-1] + dt * gamma * (I[i-1] + dt * k1_I/2)
    
    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0

t_end = 100
dt = 0.1

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, N, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using RK2')
plt.show()
