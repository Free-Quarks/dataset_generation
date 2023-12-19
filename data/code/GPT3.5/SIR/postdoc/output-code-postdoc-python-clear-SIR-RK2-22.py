import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_end, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    while t[-1] < t_end:
        t_next = t[-1] + dt
        S_next = S[-1] - dt * (beta * S[-1] * I[-1]) / N
        I_next = I[-1] + dt * (beta * S[-1] * I[-1]) / N - dt * gamma * I[-1]
        R_next = R[-1] + dt * gamma * I[-1]
        
        S.append(S_next)
        I.append(I_next)
        R.append(R_next)
        t.append(t_next)
    
    return S, I, R, t

# Example Usage
t_end = 100
dt = 0.1

beta = 0.3
gamma = 0.1
S_0 = 1000
I_0 = 1
R_0 = 0

S, I, R, t = SIR_RK2(beta, gamma, S_0, I_0, R_0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model - RK2')
plt.legend()
plt.show()
