import numpy as np
from scipy.integrate import odeint


def sidarthe(y, t, beta, sigma, tau, rho, alpha, theta, delta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E

    dSdt = -beta * S * (I + rho * A) / N
    dIdt = beta * S * (I + rho * A) / N - (1 - sigma) * alpha * I - sigma * theta * I - sigma * (1 - theta) * I
    dDdt = sigma * (1 - theta) * I
    dAdt = sigma * theta * I - (1 - tau) * delta * A - tau * delta * A
    dRdt = (1 - sigma) * alpha * I + sigma * (1 - theta) * I + (1 - tau) * delta * A
    dTdt = (1 - tau) * delta * A
    dHdt = tau * delta * A
    dEdt = rho * beta * S * (I + rho * A) / N

    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Example usage

# Initial conditions
S0 = 1000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

# Parameters
beta = 0.2
sigma = 0.1
tau = 0.2
rho = 0.3
alpha = 0.4
theta = 0.5
delta = 0.6

# Time vector
t = np.linspace(0, 10, 100)

# Solve the model
solution = odeint(sidarthe, [S0, I0, D0, A0, R0, T0, H0, E0], t, args=(beta, sigma, tau, rho, alpha, theta, delta))

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, solution[:, 0], label='S')
plt.plot(t, solution[:, 1], label='I')
plt.plot(t, solution[:, 2], label='D')
plt.plot(t, solution[:, 3], label='A')
plt.plot(t, solution[:, 4], label='R')
plt.plot(t, solution[:, 5], label='T')
plt.plot(t, solution[:, 6], label='H')
plt.plot(t, solution[:, 7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
