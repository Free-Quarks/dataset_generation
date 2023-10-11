import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, t_end, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    while t[-1] < t_end:
        S_new = S[-1] - beta * S[-1] * I[-1] / N * dt
        I_new = I[-1] + (beta * S[-1] * I[-1] / N - gamma * I[-1]) * dt
        R_new = R[-1] + gamma * I[-1] * dt
        
        t_new = t[-1] + dt
        
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        t.append(t_new)
    
    return S, I, R, t

# Example usage
S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R, t = SIR_model(S0, I0, R0, beta, gamma, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
