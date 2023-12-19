import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t0, tf, dt):
    N = S0 + I0 + R0
    t = np.arange(t0, tf + dt, dt)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
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

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t0 = 0
tf = 100
dt = 0.1

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t0, tf, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.title('SIR Model - RK2')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
