import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, t_end, dt):
    N = S0 + I0 + R0
    S = np.zeros(int(t_end/dt))
    I = np.zeros(int(t_end/dt))
    R = np.zeros(int(t_end/dt))
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, int(t_end/dt)):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dI_dt = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        dR_dt = gamma * I[i-1]
        
        S[i] = S[i-1] + (dS_dt * dt)
        I[i] = I[i-1] + (dI_dt * dt)
        R[i] = R[i-1] + (dR_dt * dt)
        
    return S, I, R


# Example usage
S0 = 900
I0 = 100
R0 = 0
beta = 0.35
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_end, dt)

t = np.linspace(0, t_end, int(t_end/dt))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
