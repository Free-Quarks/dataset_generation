import numpy as np
import matplotlib.pyplot as plt


def SIR_Euler(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    for i in range(int(t_max/dt)):
        S_new = S[i] - beta*S[i]*I[i]/N*dt
        I_new = I[i] + (beta*S[i]*I[i]/N - gamma*I[i])*dt
        R_new = R[i] + gamma*I[i]*dt
        
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        t.append((i+1)*dt)
        
    return S, I, R, t


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R, t = SIR_Euler(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
