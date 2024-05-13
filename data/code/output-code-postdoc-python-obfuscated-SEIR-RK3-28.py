import numpy as np
import matplotlib.pyplot as plt

# Function to compute the derivative of the compartments

def deriv(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Function to solve the differential equations

def seir_model(N, beta, gamma, sigma, I0, E0, R0, days):
    S0 = N - I0 - R0 - E0
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T
    return t, S, E, I, R

# Define parameters
N = 1000000
beta = 0.2
gamma = 0.1
sigma = 0.05
I0, E0, R0 = 100, 10, 0

# Solve the model
t, S, E, I, R = seir_model(N, beta, gamma, sigma, I0, E0, R0, days=100)

# Plotting
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()
