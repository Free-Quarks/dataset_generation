import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Total population, N.
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # Disease transmission rate, sigma, (in 1/days).
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0
    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # A grid of time points (in days)
    t = np.linspace(0, T, T)

    # The SEIR model differential equations.
    def deriv(y, t, N, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Integrate the SEIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T

    # Plotting the results
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model')
    plt.show()


# Parameters
gamma = 1/14
sigma = 1/5.2
beta = 0.2
days = 160
N = 1000
I0 = 1
E0 = 0
R0 = 0

seir_model(beta, gamma, sigma, N, I0, E0, R0, days)

