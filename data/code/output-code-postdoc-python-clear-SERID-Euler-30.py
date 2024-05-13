import numpy as np
import matplotlib.pyplot as plt

def serid_model(beta, gamma, N, I0, R0, t_end, dt):
    t = np.linspace(0, t_end, num=int(t_end/dt)+1)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = N - I0 - R0
    E[0] = 0
    I[0] = I0
    R[0] = R0

    for i in range(len(t) - 1):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - gamma * E[i]
        dIdt = gamma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]

        S[i+1] = S[i] + dt * dSdt
        E[i+1] = E[i] + dt * dEdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt

    return t, S, E, I, R

# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0

t_end = 100
dt = 0.1

t, S, E, I, R = serid_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SERID Model')
plt.show()
