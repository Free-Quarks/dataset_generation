import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, E0, I0, R0, t_max):
    # Initialize arrays to store the values of each compartment
    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    t = np.arange(t_max)

    # Set initial values
    S[0] = N - E0 - I0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Run the simulation using RK3 method
    for i in range(1, t_max):
        dE_dt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI_dt = sigma * E[i-1] - gamma * I[i-1]
        dR_dt = gamma * I[i-1]

        S[i] = S[i-1] - beta * S[i-1] * I[i-1] / N
        E[i] = E[i-1] + dE_dt
        I[i] = I[i-1] + dI_dt
        R[i] = R[i-1] + dR_dt

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SEIR Model Simulation')
    plt.show()

