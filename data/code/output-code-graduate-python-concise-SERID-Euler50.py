import numpy as np
import matplotlib.pyplot as plt

def serid_euler(beta, gamma, sigma, N, I0, R0, D0, t_end, dt):
    t = np.linspace(0, t_end, int(t_end/dt)+1)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    D = np.zeros_like(t)
    S[0] = N - I0 - R0 - D0
    E[0] = I0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    for i in range(1, len(t)):
        S[i] = S[i-1] - (beta*S[i-1]*I[i-1]/N)*dt
        E[i] = E[i-1] + (beta*S[i-1]*I[i-1]/N - sigma*E[i-1])*dt
        I[i] = I[i-1] + (sigma*E[i-1] - gamma*I[i-1])*dt
        R[i] = R[i-1] + (gamma*I[i-1])*dt
        D[i] = D[i-1] + (gamma*I[i-1])*dt
    return t, S, E, I, R, D

beta = 0.2
gamma = 0.1
sigma = 0.05
N = 1000
I0 = 1
R0 = 0
D0 = 0
t_end = 100
dt = 0.1

t, S, E, I, R, D = serid_euler(beta, gamma, sigma, N, I0, R0, D0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using Euler Method')
plt.legend()
plt.show()
