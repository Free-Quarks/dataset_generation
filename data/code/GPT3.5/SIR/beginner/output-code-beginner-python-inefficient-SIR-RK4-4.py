import numpy as np
import matplotlib.pyplot as plt


# Function to solve the SIR model using RK4

def solve_sir_model(beta, gamma, population, initial_infected, t_max, dt):
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    dSdt = np.zeros(n_steps)
    dIdt = np.zeros(n_steps)
    dRdt = np.zeros(n_steps)
    S[0] = population - initial_infected
    I[0] = initial_infected
    R[0] = 0

    for i in range(1, n_steps):
        dSdt[i-1] = -beta * S[i-1] * I[i-1] / population
        dIdt[i-1] = beta * S[i-1] * I[i-1] / population - gamma * I[i-1]
        dRdt[i-1] = gamma * I[i-1]

        S[i] = S[i-1] + dt * dSdt[i-1]
        I[i] = I[i-1] + dt * dIdt[i-1]
        R[i] = R[i-1] + dt * dRdt[i-1]

    return t, S, I, R


# Parameters
beta = 0.2
gamma = 0.1
population = 1000
initial_infected = 1

# Solve the SIR model
t, S, I, R = solve_sir_model(beta, gamma, population, initial_infected, t_max=100, dt=0.1)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

