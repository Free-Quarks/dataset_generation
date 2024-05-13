import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(beta, gamma, N, I0, R0, t_end, dt):
    # Define the differential equations
    def derivs(state, t):
        S, E, I, R = state
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - gamma * E
        dIdt = gamma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    # Set initial conditions
    S0 = N - I0 - R0
    E0 = 0

    # Create a time array
    t = np.arange(0, t_end, dt)

    # Create an empty array to store the solution
    solution = np.zeros((len(t), 4))

    # Set initial conditions in the solution array
    solution[0] = [S0, E0, I0, R0]

    # Apply RK2 method to solve the differential equations
    for i in range(1, len(t)):
        state_n = solution[i-1]
        t_n = t[i-1]
        k1 = derivs(state_n, t_n)
        state_n_predictor = state_n + dt * k1
        t_n_predictor = t_n + dt
        k2 = derivs(state_n_predictor, t_n_predictor)
        solution[i] = state_n + 0.5 * dt * (k1 + k2)

    # Extract the compartments from the solution array
    S, E, I, R = solution.T

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.title('SERID Model Simulation')
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
t_end = 100
dt = 0.1

serid_rk2(beta, gamma, N, I0, R0, t_end, dt)
