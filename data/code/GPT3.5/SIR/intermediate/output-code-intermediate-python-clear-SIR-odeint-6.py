import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# define the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# initial conditions
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0

# parameters
beta = 0.2
gamma = 0.1

# time points
t = np.linspace(0, 100, 100)

# solve the ODEs
y = odeint(sir_model, (S0, I0, R0), t, args=(beta, gamma))

# plot the results
plt.plot(t, y[:, 0], 'b', label='S')
plt.plot(t, y[:, 1], 'r', label='I')
plt.plot(t, y[:, 2], 'g', label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()


