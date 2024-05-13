import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def serid_model(y, t, beta, gamma, delta):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    dDdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_serid_model(S0, E0, I0, R0, D0, beta, gamma, delta, days):
    y0 = [S0, E0, I0, R0, D0]
    t = np.linspace(0, days, days)
    result = odeint(serid_model, y0, t, args=(beta, gamma, delta))
    S, E, I, R, D = result.T
    return S, E, I, R, D


S0 = 9999
E0 = 1
I0 = 0
R0 = 0
D0 = 0
beta = 0.3
gamma = 0.1
delta = 0.2
days = 100

S, E, I, R, D = run_serid_model(S0, E0, I0, R0, D0, beta, gamma, delta, days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SERID Model')
plt.legend()
plt.show()
