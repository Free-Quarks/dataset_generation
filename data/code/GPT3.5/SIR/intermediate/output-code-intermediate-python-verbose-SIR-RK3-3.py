import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, N, t_end, t_step):
    t = np.arange(0, t_end, t_step)
    num_steps = len(t)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, num_steps):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + 0.5*t_step*k1) * (I[i-1] + 0.5*t_step*k1) / N
        k3 = -beta * (S[i-1] + 0.75*t_step*k2) * (I[i-1] + 0.75*t_step*k2) / N
        
        l1 = beta * (S[i-1] * I[i-1] / N) - gamma * I[i-1]
        l2 = beta * ((S[i-1] + 0.5*t_step*k1) * (I[i-1] + 0.5*t_step*l1) / N) - gamma * (I[i-1] + 0.5*t_step*l1)
        l3 = beta * ((S[i-1] + 0.75*t_step*k2) * (I[i-1] + 0.75*t_step*l2) / N) - gamma * (I[i-1] + 0.75*t_step*l2)
        
        S[i] = S[i-1] + (t_step/9) * (2*k1 + 3*k2 + 4*k3)
        I[i] = I[i-1] + (t_step/9) * (2*l1 + 3*l2 + 4*l3)
        R[i] = R[i-1] + (t_step/9) * (2*l1 + 3*l2 + 4*l3)
    
    return S, I, R


beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0

# Time parameters
t_end = 100
t_step = 0.1

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, N, t_end, t_step)

plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.grid(True)
plt.show()
