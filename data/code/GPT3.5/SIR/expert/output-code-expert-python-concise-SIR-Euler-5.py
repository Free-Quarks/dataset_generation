import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, t_end, dt):
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dS_dt = -beta * S[i-1] * I[i-1]
        dI_dt = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dR_dt = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dS_dt
        I[i] = I[i-1] + dt * dI_dt
        R[i] = R[i-1] + dt * dR_dt
    
    return S, I, R

S0 = 1000
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
