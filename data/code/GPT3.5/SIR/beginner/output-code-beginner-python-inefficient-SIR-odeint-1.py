import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 0.1

t = np.linspace(0, 100, 100)

y0 = S0, I0, R0

result = odeint(SIR_model, y0, t, args=(N, beta, gamma))

fig, ax = plt.subplots()
ax.plot(t, result[:, 0], 'b', label='Susceptible')
ax.plot(t, result[:, 1], 'r', label='Infected')
ax.plot(t, result[:, 2], 'g', label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_title('SIR Model')
ax.legend()

plt.show()
