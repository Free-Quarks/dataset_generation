import numpy as np
import matplotlib.pyplot as plt


# Function to run the SIR model using the Runge-Kutta 2nd order method

def run_sir_model(beta, gamma, population, initial_infected, num_days):
    # Initialize arrays to store the values
    t = np.arange(num_days)
    S = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)

    # Set initial conditions
    S[0] = population - initial_infected
    I[0] = initial_infected
    R[0] = 0

    # Run the model
    for i in range(1, num_days):
        dt = t[i] - t[i-1]
        dS = -beta * S[i-1] * I[i-1] / population
        dI = beta * S[i-1] * I[i-1] / population - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()


# Example usage
run_sir_model(0.3, 0.1, 1000, 10, 100)
