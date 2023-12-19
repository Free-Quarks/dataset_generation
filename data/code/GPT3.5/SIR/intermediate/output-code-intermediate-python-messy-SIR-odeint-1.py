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


# Initial conditions
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the ODEs
sol = odeint(SIR_model, [S0, I0, R0], t, args=(N, beta, gamma))
S, I, R = sol.T

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
