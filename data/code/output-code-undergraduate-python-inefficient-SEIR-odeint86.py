import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function to define the SEIR model equations

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Initial conditions

S0 = 1000
E0 = 1
I0 = 0
R0 = 0

y0 = S0, E0, I0, R0


# Parameters

beta = 0.2
sigma = 0.1
gamma = 0.05


# Time vector

t = np.linspace(0, 100, 1000)


# Solve the SEIR model equations

solution = odeint(seir_model, y0, t, args=(beta, gamma, sigma))


# Plot the results

plt.plot(t, solution[:, 0], 'b', label='Susceptible')
plt.plot(t, solution[:, 1], 'y', label='Exposed')
plt.plot(t, solution[:, 2], 'r', label='Infected')
plt.plot(t, solution[:, 3], 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

