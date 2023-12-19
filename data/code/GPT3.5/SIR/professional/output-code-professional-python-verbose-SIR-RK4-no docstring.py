import numpy as np
import matplotlib.pyplot as plt


# Function to compute the derivatives of the SIR model

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Function to simulate and plot the SIR model


def simulate_sir_model(beta, gamma, S_0, I_0, R_0, t_max):
    # Time step size
    dt = 0.1
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps + 1)
    y = np.zeros((num_steps + 1, 3))
    y[0] = S_0, I_0, R_0

    for i in range(num_steps):
        k1 = dt * sir_model(t[i], y[i], beta, gamma)
        k2 = dt * sir_model(t[i] + 0.5 * dt, y[i] + 0.5 * k1, beta, gamma)
        k3 = dt * sir_model(t[i] + 0.5 * dt, y[i] + 0.5 * k2, beta, gamma)
        k4 = dt * sir_model(t[i] + dt, y[i] + k3, beta, gamma)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    plt.plot(t, y[:, 0], label='S')
    plt.plot(t, y[:, 1], label='I')
    plt.plot(t, y[:, 2], label='R')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()


# Example usage

simulate_sir_model(0.2, 0.1, 1000, 1, 0, 100)

