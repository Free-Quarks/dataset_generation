import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, R0, t_max):
    # Initial conditions
    S0 = N - I0 - R0
    E0 = 0
    # Create arrays to store the values
    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    # Set initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    # Set time step
    dt = 0.1

    # Run the simulation
    for t in range(1, t_max):
        # Calculate derivatives
        dS_dt = -beta * S[t-1] * I[t-1] / N
        dE_dt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dI_dt = sigma * E[t-1] - gamma * I[t-1]
        dR_dt = gamma * I[t-1]
        # Update values using RK3
        S[t] = S[t-1] + dt * dS_dt
        E[t] = E[t-1] + dt * dE_dt
        I[t] = I[t-1] + dt * dI_dt
        R[t] = R[t-1] + dt * dR_dt

    # Plot the results
    plt.figure()
    plt.plot(range(t_max), S, label='Susceptible')
    plt.plot(range(t_max), E, label='Exposed')
    plt.plot(range(t_max), I, label='Infected')
    plt.plot(range(t_max), R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


seir_model(0.3, 0.1, 0.2, 1000, 1, 0, 1000)
