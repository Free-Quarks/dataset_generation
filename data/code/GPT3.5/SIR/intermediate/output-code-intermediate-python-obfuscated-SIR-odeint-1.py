import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function implementing the SIR model

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Model parameters

N = 1000
beta = 0.2
D = 10

# Initial conditions

S0 = N - 1
I0 = 1
R0 = 0

# Time vector

t = np.linspace(0, 100, 100)

# Initial conditions vector

y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.

ret = odeint(SIR_model, y0, t, args=(N, beta, D))
S, I, R = ret.T

# Plotting the results

plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
