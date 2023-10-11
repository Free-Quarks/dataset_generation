import numpy as np
from scipy.integrate import odeint

# function that returns dy/dt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# initial conditions
y0 = [0.99, 0.01, 0.0]

# time points
t = np.linspace(0, 100, 100)

# parameters
gamma = 0.1
gamma = 1/gamma
beta = 0.2

# solve ODE
y = odeint(SIR_model, y0, t, args=(beta, gamma))

# plot result
import matplotlib.pyplot as plt

plt.plot(t, y[:, 0], 'b', label='Susceptible')
plt.plot(t, y[:, 1], 'r', label='Infected')
plt.plot(t, y[:, 2], 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SIR Model')
plt.legend()
plt.show()
