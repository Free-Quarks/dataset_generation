import numpy as np
import matplotlib.pyplot as plt

# Function to define the SIR model


# Function to perform Runge-Kutta 4th order method


def simulate_sir_model(N, beta, gamma, I0, R0, t_end, dt):
    def sir_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def runge_kutta(t, y, dt):
        k1 = dt * sir_model(t, y)
        k2 = dt * sir_model(t + 0.5 * dt, y + 0.5 * k1)
        k3 = dt * sir_model(t + 0.5 * dt, y + 0.5 * k2)
        k4 = dt * sir_model(t + dt, y + k3)
        return y + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        y = np.array([S[i - 1], I[i - 1], R[i - 1]])
        y = runge_kutta(t[i - 1], y, dt)
        S[i] = y[0]
        I[i] = y[1]
        R[i] = y[2]

    return t, S, I, R


# Simulation parameters
N = 100000  # Total population
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
I0 = 100  # Initial infected
R0 = 0  # Initial recovered
t_end = 100  # Time period
dt = 0.1  # Time step

# Run simulation
t, S, I, R = simulate_sir_model(N, beta, gamma, I0, R0, t_end, dt)

# Plotting the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
