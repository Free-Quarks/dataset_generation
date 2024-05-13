import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SERID_RK3(y, t, N, beta, gamma, delta):
    S, E, R, I, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dRdt = gamma * I
    dIdt = delta * E - gamma * I
    dDdt = delta * E
    return [dSdt, dEdt, dRdt, dIdt, dDdt]

N = 1000
E0, I0, R0, D0 = 1, 0, 0, 0
S0 = N - E0 - I0 - R0 - D0
beta, gamma, delta = 0.2, 1/10, 1/20
y0 = [S0, E0, R0, I0, D0]
t = np.linspace(0, 160, 160)

solution = odeint(SERID_RK3, y0, t, args=(N, beta, gamma, delta))

S, E, R, I, D = solution.T

plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, D, 'k', label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SERID Model using RK3')
plt.legend()
plt.grid(True)
plt.show()
