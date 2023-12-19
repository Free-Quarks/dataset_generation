import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]


def plot_results(t, S, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.title('SIR Model')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()


# Initial conditions
S0 = 999
I0 = 1
R0 = 0
y0 = [S0, I0, R0]

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the ODEs
solution = odeint(SIR_model, y0, t, args=(beta, gamma))

# Extract the solutions
S = solution[:, 0]
I = solution[:, 1]
R = solution[:, 2]

# Plot the results
plot_results(t, S, I, R)
