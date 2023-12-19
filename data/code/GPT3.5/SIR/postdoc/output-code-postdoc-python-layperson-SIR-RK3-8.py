import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, N, I0, R0, t_max):
    dt = 0.1
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)

    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for i in range(1, num_steps):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    return t, S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_max = 10

# Simulate SIR model
t, S, I, R = sir_model(beta, gamma, N, I0, R0, t_max)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
