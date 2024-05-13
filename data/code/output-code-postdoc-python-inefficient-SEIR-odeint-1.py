import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(initial_state, t, beta, sigma, gamma):
    # Integrate the SEIR equations over the time grid t
    sol = odeint(seir_model, initial_state, t, args=(beta, sigma, gamma))
    S, E, I, R = sol.T

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


# Define the initial conditions
initial_state = [9999, 1, 0, 0]  # S, E, I, R

# Define the time grid
t = np.linspace(0, 100, 100)

# Define the parameters
beta = 0.2
sigma = 0.1
gamma = 0.05

# Run the SEIR model
run_seir_model(initial_state, t, beta, sigma, gamma)
