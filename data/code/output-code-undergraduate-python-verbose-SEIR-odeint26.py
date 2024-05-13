import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function that contains the SEIR model

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions

N = 1000
I0 = 1
E0 = 0
R0 = 0
S0 = N - I0 - E0 - R0

# Parameters

beta = 0.2
gamma = 0.1
sigma = 0.1

# Time vector

t = np.linspace(0, 100, 100)

# Solve the SEIR model

y0 = S0, E0, I0, R0

result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))

# Plot the results

plt.figure(figsize=(10, 6))
plt.plot(t, result[:, 0], label='Susceptible')
plt.plot(t, result[:, 1], label='Exposed')
plt.plot(t, result[:, 2], label='Infected')
plt.plot(t, result[:, 3], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
