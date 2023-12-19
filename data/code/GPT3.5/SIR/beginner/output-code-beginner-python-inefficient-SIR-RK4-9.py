import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the derivatives

def derivs(state, t, beta, gamma):
    S, I, R = state
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to solve the ODE using RK4

def solve_sir_rk4(init_state, t, beta, gamma):
    states = [init_state]
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = derivs(states[i-1], t[i-1], beta, gamma)
        k2 = derivs(states[i-1] + 0.5 * dt * k1, t[i-1] + 0.5 * dt, beta, gamma)
        k3 = derivs(states[i-1] + 0.5 * dt * k2, t[i-1] + 0.5 * dt, beta, gamma)
        k4 = derivs(states[i-1] + dt * k3, t[i-1] + dt, beta, gamma)
        next_state = states[i-1] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(next_state)
    return np.array(states)

# Parameter values
beta = 0.6
gamma = 0.1

# Initial conditions
init_state = [0.99, 0.01, 0.0]

# Time points
t = np.linspace(0, 100, 1000)

# Solve SIR model
states = solve_sir_rk4(init_state, t, beta, gamma)

# Plotting
plt.plot(t, states[:, 0], label='Susceptible')
plt.plot(t, states[:, 1], label='Infected')
plt.plot(t, states[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
