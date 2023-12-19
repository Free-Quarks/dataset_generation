import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, S0, I0, R0, t_end, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_end, dt)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        S1 = S[i-1] + dt * dSdt
        I1 = I[i-1] + dt * dIdt
        R1 = R[i-1] + dt * dRdt
        
        dSdt1 = -beta * S1 * I1 / N
        dIdt1 = beta * S1 * I1 / N - gamma * I1
        dRdt1 = gamma * I1
        
        S2 = (3/4) * S[i-1] + (1/4) * S1 + (1/4) * dt * dSdt1
        I2 = (3/4) * I[i-1] + (1/4) * I1 + (1/4) * dt * dIdt1
        R2 = (3/4) * R[i-1] + (1/4) * R1 + (1/4) * dt * dRdt1
        
        S[i] = (1/3) * S[i-1] + (2/3) * S2 + (2/3) * dt * dSdt1
        I[i] = (1/3) * I[i-1] + (2/3) * I2 + (2/3) * dt * dIdt1
        R[i] = (1/3) * R[i-1] + (2/3) * R2 + (2/3) * dt * dRdt1
    
    return S, I, R


beta = 0.25
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_end = 100
dt = 0.1

S, I, R = sir_model(beta, gamma, S0, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
