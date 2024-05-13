import numpy as np
import matplotlib.pyplot as plt

def SEIR_RK3(N, beta, gamma, sigma, E0, I0, R0, t_final, dt):
    num_steps = int(t_final/dt)
    t = np.linspace(0, t_final, num_steps)
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = N - E0 - I0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for i in range(1, num_steps):
        dSdt = -beta*S[i-1]*I[i-1]/N
        dEdt = beta*S[i-1]*I[i-1]/N - sigma*E[i-1]
        dIdt = sigma*E[i-1] - gamma*I[i-1]
        dRdt = gamma*I[i-1]
        S[i] = S[i-1] + dt*dSdt
        E[i] = E[i-1] + dt*dEdt
        I[i] = I[i-1] + dt*dIdt
        R[i] = R[i-1] + dt*dRdt
    return S, E, I, R

N = 10000
beta = 0.25
gamma = 0.1
sigma = 0.2
E0 = 10
I0 = 1
R0 = 0
t_final = 100
dt = 0.1

S, E, I, R = SEIR_RK3(N, beta, gamma, sigma, E0, I0, R0, t_final, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

