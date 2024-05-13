import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def serid_model(y, t, beta, gamma, alpha):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dRdt = gamma * I
    dIdt = alpha * E - gamma * I
    dDdt = gamma * I
    return dSdt, dEdt, dRdt, dIdt, dDdt


def simulate_serid_model(S0, E0, R0, I0, D0, beta, gamma, alpha, t):
    y0 = S0, E0, R0, I0, D0
    params = beta, gamma, alpha
    return odeint(serid_model, y0, t, args=params)


# Set parameters
S0 = 990
E0 = 10
R0 = 0
I0 = 0
D0 = 0
beta = 0.3
gamma = 0.1
alpha = 0.05
T = 100

# Time vector
t = np.linspace(0, T, T+1)

# Simulate the SEIR model
y = simulate_serid_model(S0, E0, R0, I0, D0, beta, gamma, alpha, t)

# Plot the results
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Exposed')
plt.plot(t, y[:, 2], label='Recovered')
plt.plot(t, y[:, 3], label='Infected')
plt.plot(t, y[:, 4], label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model Simulation')
plt.legend()
plt.show()
