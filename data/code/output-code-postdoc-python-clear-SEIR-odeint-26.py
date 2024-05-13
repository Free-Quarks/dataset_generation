import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that returns dy/dt

def SEIR(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Total population, N

N = 1000

# Initial number of infected and recovered individuals, I0 and R0

I0, R0 = 1, 0

# Everyone else, S0, is susceptible to infection initially

S0 = N - I0 - R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days)

beta, gamma = 0.2, 1./10

# A rate describing how many people one infected person infects per day, sigma

sigma = 1./5

# A grid of time points (in days)

t = np.linspace(0, 160, 160)

# Initial conditions vector

y0 = S0, 1, I0, R0

# Integrate the SEIR equations over the time grid, t

ret = odeint(SEIR, y0, t, args=(N, beta, gamma, sigma))

S, E, I, R = ret.T

# Plotting the data

fig, ax = plt.subplots()
ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of individuals')
ax.set_title('SEIR Model')
ax.legend(loc='best')
plt.show()
