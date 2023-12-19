import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, t_max, dt):
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dI_dt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR_dt = gamma * I[i-1]

        k1_S = dt * dS_dt
        k1_I = dt * dI_dt
        k1_R = dt * dR_dt
        k2_S = dt * (-beta * (S[i-1] + k1_S/2) * (I[i-1] + k1_I/2) / N)
        k2_I = dt * (beta * (S[i-1] + k1_S/2) * (I[i-1] + k1_I/2) / N - gamma * (I[i-1] + k1_I/2))
        k2_R = dt * gamma * (I[i-1] + k1_I/2)
        k3_S = dt * (-beta * (S[i-1] - k1_S + 2*k2_S) * (I[i-1] - k1_I + 2*k2_I) / N)
        k3_I = dt * (beta * (S[i-1] - k1_S + 2*k2_S) * (I[i-1] - k1_I + 2*k2_I) / N - gamma * (I[i-1] - k1_I + 2*k2_I))
        k3_R = dt * gamma * (I[i-1] - k1_I + 2*k2_I)

        S[i] = S[i-1] + (1/6) * (k1_S + 4*k2_S + k3_S)
        I[i] = I[i-1] + (1/6) * (k1_I + 4*k2_I + k3_I)
        R[i] = R[i-1] + (1/6) * (k1_R + 4*k2_R + k3_R)

    return t, S, I, R


# Example usage

# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0

t_max = 100
dt = 0.1

# Run the model
t, S, I, R = sir_model(beta, gamma, N, I0, R0, t_max, dt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()
