import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe(y, t, N, alpha, beta, gamma, delta, epsilon, theta, rho, k):
    S, I, D, A, R, T, H, E = y
    dSdt = -alpha * S * (I + rho * D + theta * A) / N
    dIdt = alpha * S * (I + rho * D + theta * A) / N - beta * I - delta * I - epsilon * I
    dDdt = delta * I - gamma * D
    dAdt = epsilon * I - gamma * A
    dRdt = beta * I + rho * D + theta * A - gamma * R
    dTdt = k * delta * I - gamma * T
    dHdt = (1 - k) * delta * I - gamma * H
    dEdt = alpha * S * (I + rho * D + theta * A) / N
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


N = 1000000
alpha = 0.4
beta = 0.2
gamma = 0.1
delta = 0.05
epsilon = 0.01
theta = 0.01
rho = 0.01
k = 0.8
T = 300

t = np.linspace(0, T, T)
y0 = N-1, 1, 0, 0, 0, 0, 0, 0

ret = odeint(sidarthe, y0, t, args=(N, alpha, beta, gamma, delta, epsilon, theta, rho, k))
S, I, D, A, R, T, H, E = ret.T


plt.figure(figsize=(10,6))
plt.plot(t, S/1000, 'b', label='Susceptible')
plt.plot(t, I/1000, 'r', label='Infected')
plt.plot(t, D/1000, 'g', label='Deceased')
plt.plot(t, A/1000, 'y', label='Asymptomatic')
plt.plot(t, R/1000, 'm', label='Recovered')
plt.plot(t, T/1000, 'c', label='Tested')
plt.plot(t, H/1000, 'k', label='Hospitalized')
plt.plot(t, E/1000, 'purple', label='Exposed')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals (thousands)')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
