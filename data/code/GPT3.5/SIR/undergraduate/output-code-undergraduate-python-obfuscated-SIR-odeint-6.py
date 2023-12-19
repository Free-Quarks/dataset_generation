import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt

def model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# initial conditions
define your initial conditions here
N = ...
S0 = ...
I0 = ...
R0 = ...

# parameters
define your parameters here
beta = ...
gamma = ...

# time points
t = np.linspace(0, ..., num=...)

# initial condition vector
y0 = S0, I0, R0

# solve ODE
y = odeint(model, y0, t, args=(N, beta, gamma))
S, I, R = y.T

# plot results
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()


