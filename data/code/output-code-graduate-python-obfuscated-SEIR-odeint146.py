import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function implementing the SEIR model
def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions and parameters
N = 100000
beta = 0.2
sigma = 1/5
gamma = 1/10
S0, E0, I0, R0 = N-1, 1, 0, 0

# Time vector
t = np.linspace(0, 100, 100)

# Solve the SEIR model
y0 = S0, E0, I0, R0
result = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = result.T

# Plot the SEIR model
def plot_seir_model(t, S, E, I, R):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_ylim(0, N)
    ax.set_title('SEIR Model')
    ax.legend()
    plt.show()

plot_seir_model(t, S, E, I, R)
