import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_rk3(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

N = 1000
beta = 1.0
gamma = 0.1

y0 = [N-1, 1, 0]

t = np.linspace(0, 200, 100)
sol = solve_ivp(sir_rk3, [0, 200], y0, t_eval=t, args=(N, beta, gamma))

plt.plot(t, sol.y[0], 'b', label='Susceptible')
plt.plot(t, sol.y[1], 'r', label='Infected')
plt.plot(t, sol.y[2], 'g', label='Recovered')
plt.legend()
plt.show()
