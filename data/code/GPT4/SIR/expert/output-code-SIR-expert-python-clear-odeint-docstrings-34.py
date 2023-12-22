import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

S0 = 0.99
I0 = 0.01
R0 = 0.0
beta = 0.35
gamma = 0.1

t = np.linspace(0, 100, 10000)

sol = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))

plt.figure(figsize=[6, 4])
plt.plot(t, sol[:, 0], label='Susceptible')
plt.plot(t, sol[:, 1], label='Infected')
plt.plot(t, sol[:, 2], label='Recovered')
plt.legend()
plt.show()
