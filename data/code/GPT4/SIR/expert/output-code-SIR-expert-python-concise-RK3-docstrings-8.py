import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_model(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 1./10 

y0 = S0, I0, R0

ret = solve_ivp(sir_model, (0, 160), y0, args=(N, beta, gamma),
                dense_output=True)

t = np.linspace(0, 160, 160)
S, I, R = ret.sol(t)

plt.figure(figsize=(6,4))
plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.legend()
plt.show()
