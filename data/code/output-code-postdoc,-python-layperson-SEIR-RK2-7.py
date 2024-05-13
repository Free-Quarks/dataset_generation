import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Define the SEIR model
    def deriv(y, t, beta, gamma, sigma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Set up the initial conditions
    y0 = N - I0 - E0 - R0
    t = np.linspace(0, T, T)
    # Integrate the SEIR equations over the time grid, t
    ret = odeint(deriv, [S0, E0, I0, R0], t, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T

    # Plot the data
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model Simulation')
    plt.show()


seir_model(0.2, 1/14, 1/5, 1000, 1, 0, 0, 100)

