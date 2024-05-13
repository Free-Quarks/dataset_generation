import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def serid_model(y, t, beta, gamma, delta):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dRdt = gamma * I
    dIdt = delta * E - gamma * I
    dDdt = delta * E
    return [dSdt, dEdt, dRdt, dIdt, dDdt]


def serid_simulation(S0, E0, R0, I0, D0, beta, gamma, delta, t):
    y0 = [S0, E0, R0, I0, D0]
    sol = odeint(serid_model, y0, t, args=(beta, gamma, delta))
    S, E, R, I, D = sol.T
    return S, E, R, I, D


# Example usage
S0 = 999
E0 = 1
R0 = 0
I0 = 0
D0 = 0
beta = 0.3
gamma = 0.1
delta = 0.05
t = np.linspace(0, 100, 1000)
S, E, R, I, D = serid_simulation(S0, E0, R0, I0, D0, beta, gamma, delta, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, R, label='Recovered')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
