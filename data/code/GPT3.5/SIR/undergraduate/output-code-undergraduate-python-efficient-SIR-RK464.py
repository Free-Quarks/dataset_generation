import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        l1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = -beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * l1) / N
        l2 = beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * l1) / N - gamma * (I[i-1] + 0.5 * dt * l1)
        k3 = -beta * (S[i-1] + 0.5 * dt * k2) * (I[i-1] + 0.5 * dt * l2) / N
        l3 = beta * (S[i-1] + 0.5 * dt * k2) * (I[i-1] + 0.5 * dt * l2) / N - gamma * (I[i-1] + 0.5 * dt * l2)
        k4 = -beta * (S[i-1] + dt * k3) * (I[i-1] + dt * l3) / N
        l4 = beta * (S[i-1] + dt * k3) * (I[i-1] + dt * l3) / N - gamma * (I[i-1] + dt * l3)
        S[i] = S[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        I[i] = I[i-1] + (dt/6) * (l1 + 2*l2 + 2*l3 + l4)
        R[i] = N - S[i] - I[i]

    return t, S, I, R

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Simulation of SIR Model using RK4')
plt.legend()
plt.show()
