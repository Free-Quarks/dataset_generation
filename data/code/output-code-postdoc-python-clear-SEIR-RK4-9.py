import numpy as np
import matplotlib.pyplot as plt


# Function to define the SEIR model

def SEIR_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Function to solve the SEIR model using RK4

def solve_SEIR_model(N, beta, gamma, sigma, E0, I0, R0, T):
    t = np.linspace(0, T, T+1)
    y0 = N - E0 - I0 - R0
    S0, E0, I0, R0 = y0, E0, I0, R0
    y = S0, E0, I0, R0
    solution = np.zeros((T+1, 4))
    solution[0] = y
    for i in range(T):
        h = t[i+1] - t[i]
        k1 = h * SEIR_model(y, t[i], N, beta, gamma, sigma)
        k2 = h * SEIR_model(y + 0.5 * k1, t[i] + 0.5 * h, N, beta, gamma, sigma)
        k3 = h * SEIR_model(y + 0.5 * k2, t[i] + 0.5 * h, N, beta, gamma, sigma)
        k4 = h * SEIR_model(y + k3, t[i+1], N, beta, gamma, sigma)
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        solution[i+1] = y
    return solution


# Parameters

N = 1000000  # Population size
beta = 0.2  # Transmission rate
gamma = 0.1  # Recovery rate
sigma = 0.5  # Incubation rate
E0 = 10  # Initial exposed
I0 = 1  # Initial infected
R0 = 0  # Initial recovered
T = 100  # Time period

solution = solve_SEIR_model(N, beta, gamma, sigma, E0, I0, R0, T)

# Plotting

plt.plot(solution[:, 0], label='Susceptible')
plt.plot(solution[:, 1], label='Exposed')
plt.plot(solution[:, 2], label='Infected')
plt.plot(solution[:, 3], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model Simulation')
plt.legend()
plt.show()
