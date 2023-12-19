import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_end, dt):
    t = np.arange(0, t_end, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + 0.5*dt*k1) * (I[i-1] + 0.5*dt*k1) / N
        k3 = -beta * (S[i-1] + 0.75*dt*k2) * (I[i-1] + 0.75*dt*k2) / N
        S[i] = S[i-1] + dt * (k1/3 + k2/6 + k3/6)
        I[i] = I[i-1] + dt * (k1/3 + k2/6 + k3/6)
        R[i] = R[i-1] + dt * (gamma * I[i-1] + dt * gamma * (k1/3 + k2/6 + k3/6))
        
    return S, I, R

beta = 0.3
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

