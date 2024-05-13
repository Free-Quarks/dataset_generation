import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def plot_seir(t, S, E, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


# Parameters
t_max = 100
N = 1000
beta = 0.2
sigma = 1/5
gamma = 1/10

# Initial conditions
S0, E0, I0, R0 = N-1, 1, 0, 0

# Time vector
t = np.linspace(0, t_max, t_max+1)

# Solve the SEIR model
sol = odeint(seir_model, (S0, E0, I0, R0), t, args=(N, beta, sigma, gamma))

# Unpack the solution
S, E, I, R = sol.T

# Plot the results
plot_seir(t, S, E, I, R)
