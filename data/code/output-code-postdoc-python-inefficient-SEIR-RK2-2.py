import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, N, I0, R0, E0, S0, t_end):
    # Define the compartmental model
    def seir_deriv(y, t, beta, gamma, sigma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Set up initial conditions
    y0 = S0, E0, I0, R0

    # Set up time grid
    t = np.linspace(0, t_end, t_end+1)

    # Integrate the SEIR equations over the time grid
    ret = odeint(seir_deriv, y0, t, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, E, label='Exposed')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SEIR Model')
    ax.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 1000
I0 = 1
R0 = 0
E0 = 0
S0 = N - I0 - R0 - E0
t_end = 100

seir_model(beta, gamma, sigma, N, I0, R0, E0, S0, t_end)
