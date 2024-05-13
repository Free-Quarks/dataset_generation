import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that defines the SEIR model
def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def simulate_seir_model(N, E0, I0, R0, beta, gamma, sigma, t_max):
    # Total population
    S0 = N - E0 - I0 - R0
    # Initial conditions
    y0 = S0, E0, I0, R0
    # Time vector
    t = np.linspace(0, t_max, t_max)
    # Solve the ODE system
    solution = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = solution.T
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.title('SEIR Model Simulation')
    plt.show()


# Example usage
def main():
    N = 1000
    E0 = 1
    I0 = 1
    R0 = 0
    beta = 0.2
    gamma = 0.1
    sigma = 0.05
    t_max = 100
    simulate_seir_model(N, E0, I0, R0, beta, gamma, sigma, t_max)


if __name__ == '__main__':
    main()
