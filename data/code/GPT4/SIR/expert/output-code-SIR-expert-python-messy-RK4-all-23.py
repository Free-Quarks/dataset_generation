import numpy as np
import matplotlib.pyplot as plt

# Define the model parameters
beta = 0.2
gamma = 0.1

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

def rk4(y, t, dt, f):
    k1 = f(y, t)
    k2 = f(y + dt/2 * k1, t + dt/2)
    k3 = f(y + dt/2 * k2, t + dt/2)
    k4 = f(y + dt * k3, t + dt)
    dy = dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y + dy

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.00
y0 = np.array([S0, I0, R0])

# Time vector
t = np.linspace(0, 100, 10000)

# Solve the SIR model
SIR = np.empty((3, len(t)))
SIR[:, 0] = y0

for i in range(1, len(t)):
    SIR[:, i] = rk4(SIR[:, i-1], t[i-1], t[1]-t[0], sir_model)

# Plot the data
plt.figure(figsize=[6, 4])
plt.plot(t, SIR[0, :], label='Susceptible')
plt.plot(t, SIR[1, :], label='Infected')
plt.plot(t, SIR[2, :], label='Recovered')
plt.legend()
plt.grid()
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR model using RK4 method')
plt.show()
