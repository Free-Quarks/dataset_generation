import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, T):
    dt = 0.1
    num_steps = int(T/dt)
    t = np.linspace(0, T, num_steps+1)
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(num_steps):
        k1_S = -beta*S[i]*I[i]/N
        k1_I = beta*S[i]*I[i]/N - gamma*I[i]
        k2_S = -beta*(S[i]+0.5*dt*k1_S)*(I[i]+0.5*dt*k1_I)/N
        k2_I = beta*(S[i]+0.5*dt*k1_S)*(I[i]+0.5*dt*k1_I)/N - gamma*(I[i]+0.5*dt*k1_I)
        S[i+1] = S[i] + dt*k2_S
        I[i+1] = I[i] + dt*k2_I
        R[i+1] = R[i] + dt*gamma*(I[i]+0.5*dt*k1_I)
    
    return t, S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100

t, S, I, R = SIR_RK2(beta, gamma, N, I0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
