import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(N, I0, R0, beta, gamma, t_max, dt):
    S0 = N - I0 - R0
    t = np.arange(0, t_max + dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        S[i] = S[i-1] - (beta*S[i-1]*I[i-1])*dt
        I[i] = I[i-1] + (beta*S[i-1]*I[i-1] - gamma*I[i-1])*dt
        R[i] = R[i-1] + gamma*I[i-1]*dt

    return S, I, R

N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK3(N, I0, R0, beta, gamma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
