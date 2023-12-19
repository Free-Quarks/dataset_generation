import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T):
    dt = 0.01
    nt = int(T/dt)
    t = np.linspace(0, T, nt)
    S = np.zeros(nt)
    I = np.zeros(nt)
    R = np.zeros(nt)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    for i in range(nt-1):
        k1_S = -beta*S[i]*I[i]/N
        k1_I = beta*S[i]*I[i]/N - gamma*I[i]
        k1_R = gamma*I[i]
        k2_S = -beta*(S[i] + 0.5*dt*k1_S)*(I[i] + 0.5*dt*k1_I)/N
        k2_I = beta*(S[i] + 0.5*dt*k1_S)*(I[i] + 0.5*dt*k1_I)/N - gamma*(I[i] + 0.5*dt*k1_I)
        k2_R = gamma*(I[i] + 0.5*dt*k1_I)
        k3_S = -beta*(S[i] - dt*k1_S + 2*dt*k2_S)*(I[i] - dt*k1_I + 2*dt*k2_I)/N
        k3_I = beta*(S[i] - dt*k1_S + 2*dt*k2_S)*(I[i] - dt*k1_I + 2*dt*k2_I)/N - gamma*(I[i] - dt*k1_I + 2*dt*k2_I)
        k3_R = gamma*(I[i] - dt*k1_I + 2*dt*k2_I)
        S[i+1] = S[i] + (dt/6)*(k1_S + 4*k2_S + k3_S)
        I[i+1] = I[i] + (dt/6)*(k1_I + 4*k2_I + k3_I)
        R[i+1] = R[i] + (dt/6)*(k1_R + 4*k2_R + k3_R)
    return t, S, I, R


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 10

t, S, I, R = SIR_RK3(beta, gamma, N, I0, T)

plt.plot(t, S)
plt.plot(t, I)
plt.plot(t, R)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend(['Susceptible', 'Infected', 'Recovered'])
plt.show()
