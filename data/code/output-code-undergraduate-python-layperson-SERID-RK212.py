import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(N, I0, R0, beta, gamma, sigma, t_max, dt):
    num_steps = int(t_max/dt)
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    S[0] = N - I0 - R0
    E[0] = 0
    I[0] = I0
    R[0] = R0
    
    for step in range(1, num_steps):
        S[step] = S[step-1] - dt*beta*S[step-1]*I[step-1]/N
        E[step] = E[step-1] + dt*(beta*S[step-1]*I[step-1]/N - sigma*E[step-1])
        I[step] = I[step-1] + dt*(sigma*E[step-1] - gamma*I[step-1])
        R[step] = R[step-1] + dt*gamma*I[step-1]
    
    return S, E, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.4
gamma = 0.1
sigma = 0.2
t_max = 100
dt = 0.1

S, E, I, R = serid_rk2(N, I0, R0, beta, gamma, sigma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Population size')
plt.legend()
plt.title('SERID Model using RK2')
plt.show()
