import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, N, I0, t_max, dt):
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + 0.5*k1) * (I[i-1] + 0.5*k1) / N)
        k3 = dt * (-beta * (S[i-1] - k1 + 2*k2) * (I[i-1] - k1 + 2*k2) / N)
        S[i] = S[i-1] + (1/6) * (k1 + 4*k2 + k3)

        k1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = dt * (beta * (S[i-1] + 0.5*k1) * (I[i-1] + 0.5*k1) / N - gamma * (I[i-1] + 0.5*k1))
        k3 = dt * (beta * (S[i-1] - k1 + 2*k2) * (I[i-1] - k1 + 2*k2) / N - gamma * (I[i-1] - k1 + 2*k2))
        I[i] = I[i-1] + (1/6) * (k1 + 4*k2 + k3)

        R[i] = N - S[i] - I[i]

    return t, S, I, R


N = 1000
I0 = 1
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1


# Run the model
t, S, I, R = sir_model(beta, gamma, N, I0, t_max, dt)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
