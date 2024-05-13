import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def plot_seir_model(t, S, E, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()


# Initial conditions
S0 = 9999
E0 = 1
I0 = 0
R0 = 0
y0 = S0, E0, I0, R0

# Parameters
beta = 0.35
sigma = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Integrate the SEIR equations
result = odeint(seir_model, y0, t, args=(beta, gamma, sigma))

# Unpack the solution
S, E, I, R = result.T

# Plot the SEIR model
plot_seir_model(t, S, E, I, R)
