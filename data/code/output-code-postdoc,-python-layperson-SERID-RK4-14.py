import numpy as np
import matplotlib.pyplot as plt

def serid_model(beta, gamma, N, I_init, R_init, D_init, t_max):
    S_init = N - I_init - R_init - D_init
    dt = 0.1
    total_steps = int(t_max / dt)
    t = np.linspace(0, t_max, total_steps)
    S = np.zeros(total_steps)
    I = np.zeros(total_steps)
    R = np.zeros(total_steps)
    D = np.zeros(total_steps)

    S[0] = S_init
    I[0] = I_init
    R[0] = R_init
    D[0] = D_init

    for i in range(total_steps - 1):
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]
        dDdt = 0

        S[i + 1] = S[i] + dt * dSdt
        I[i + 1] = I[i] + dt * dIdt
        R[i + 1] = R[i] + dt * dRdt
        D[i + 1] = D[i] + dt * dDdt

    return S, I, R, D


beta = 0.2
gamma = 0.1
N = 1000
I_init = 10
R_init = 0
D_init = 0
t_max = 10

S, I, R, D = serid_model(beta, gamma, N, I_init, R_init, D_init, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SERID Model Simulation')
plt.legend()
plt.show()

