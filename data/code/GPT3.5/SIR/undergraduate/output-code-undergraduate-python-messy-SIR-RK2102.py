import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, R0, t_end, dt):
    # Initial conditions
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    # RK2 method
    while t[-1] < t_end:
        S_last = S[-1]
        I_last = I[-1]
        R_last = R[-1]
        
        dS = -beta * S_last * I_last / N
        dI = beta * S_last * I_last / N - gamma * I_last
        dR = gamma * I_last
        
        S_star = S_last + dt * dS
        I_star = I_last + dt * dI
        R_star = R_last + dt * dR
        
        dS_star = -beta * S_star * I_star / N
        dI_star = beta * S_star * I_star / N - gamma * I_star
        dR_star = gamma * I_star
        
        S.append(S_last + dt/2 * (dS + dS_star))
        I.append(I_last + dt/2 * (dI + dI_star))
        R.append(R_last + dt/2 * (dR + dR_star))
        t.append(t[-1] + dt)
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R = SIR_RK2(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation using RK2')
plt.show()
