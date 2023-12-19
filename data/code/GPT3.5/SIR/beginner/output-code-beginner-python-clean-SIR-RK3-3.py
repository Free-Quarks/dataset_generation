import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, dt, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    for i in range(int(T/dt)):
        t.append((i+1)*dt)
        S_star = S[-1] - beta*S[-1]*I[-1]*dt/2
        I_star = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt/2
        R_star = R[-1] + gamma*I[-1]*dt/2
        
        S.append(S[-1] - beta*S_star*I_star*dt)
        I.append(I[-1] + (beta*S_star*I_star - gamma*I_star)*dt)
        R.append(R[-1] + gamma*I_star*dt)
        
    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0

dt = 0.1
T = 100

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, dt, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population fraction')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
