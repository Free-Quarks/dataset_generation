import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sidarthe(y, t, N, alpha, beta, delta, gamma, theta, epsilon, rho):
    S, I, D, A, R, T, H, E = y
    dSdt = -alpha * S * (I + theta * A) / N
    dIdt = alpha * S * (I + theta * A) / N - (delta + epsilon) * I
    dDdt = delta * I
    dAdt = epsilon * I - (gamma + rho) * A
    dRdt = gamma * A
    dTdt = theta * alpha * S * (I + theta * A) / N
    dHdt = rho * A
    dEdt = epsilon * I
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


N = 100000
alpha = 0.2
beta = 0.4
delta = 0.1
gamma = 0.1
theta = 0.1
epsilon = 0.01
rho = 0.01
duration = 365

S0 = N - 1
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

y0 = S0, I0, D0, A0, R0, T0, H0, E0

t = np.linspace(0, duration, duration)
solution = odeint(sidarthe, y0, t, args=(N, alpha, beta, delta, gamma, theta, epsilon, rho))
S, I, D, A, R, T, H, E = solution.T

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, S, 'b', label='Susceptible')
ax.plot(t, I, 'r', label='Infected')
ax.plot(t, D, 'g', label='Deceased')
ax.plot(t, A, 'y', label='Asymptomatic')
ax.plot(t, R, 'm', label='Recovered')
ax.plot(t, T, 'k', label='Tested')
ax.plot(t, H, 'c', label='Hospitalized')
ax.plot(t, E, 'purple', label='Exposed')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of individuals')
ax.set_title('SIDARTHE Model')
ax.legend()
plt.show()
