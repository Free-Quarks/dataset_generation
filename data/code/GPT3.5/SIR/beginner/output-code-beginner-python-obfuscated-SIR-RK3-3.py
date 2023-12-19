import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(N, beta, gamma, I0, R0, t_end):
    dt = 0.01
    t = np.linspace(0, t_end, int(t_end/dt)+1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1 = -beta*S[i-1]*I[i-1]/N
        k2 = -beta*(S[i-1]+0.5*dt*k1)*(I[i-1]+0.5*dt*k1)/N
        k3 = -beta*(S[i-1]-dt*k1+2*dt*k2)*(I[i-1]-dt*k1+2*dt*k2)/N
        S[i] = S[i-1] + dt*(k1+4*k2+k3)/6
        k1 = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
        k2 = beta*(S[i-1]+0.5*dt*k1)*(I[i-1]+0.5*dt*k1)/N - gamma*(I[i-1]+0.5*dt*k1)
        k3 = beta*(S[i-1]-dt*k1+2*dt*k2)*(I[i-1]-dt*k1+2*dt*k2)/N - gamma*(I[i-1]-dt*k1+2*dt*k2)
        I[i] = I[i-1] + dt*(k1+4*k2+k3)/6
        k1 = gamma*I[i-1]
        k2 = gamma*(I[i-1]+0.5*dt*k1)
        k3 = gamma*(I[i-1]-dt*k1+2*dt*k2)
        R[i] = R[i-1] + dt*(k1+4*k2+k3)/6
    return S, I, R

N = 1000
beta = 0.3
gamma = 0.1
I0 = 1
R0 = 0
t_end = 100
S, I, R = SIR_RK3(N, beta, gamma, I0, R0, t_end)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.title('SIR Model using RK3')
plt.show()

