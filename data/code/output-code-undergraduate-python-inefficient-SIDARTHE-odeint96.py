import numpy as np
from scipy.integrate import odeint


def sidarthe(y, t, beta, gamma, delta, alpha, mu, eta, rho, theta):
    S, E, I, A, R, T, H, E_s, D, H_s = y
    N = S + E + I + A + R + T + H + E_s + D + H_s
    dSdt = -beta * S * (I + R + T + H + E_s) / N
    dEdt = beta * S * (I + R + T + H + E_s) / N - delta * E
    dIdt = delta * (1 - alpha) * E - gamma * I
    dAdt = delta * alpha * E - mu * A
    dRdt = gamma * (1 - eta) * I
    dTdt = gamma * eta * I - theta * T
    dHdt = mu * A
    dE_sdt = delta * (1 - alpha) * E
    dDdt = theta * T
    dH_sdt = gamma * (1 - eta) * I
    return [dSdt, dEdt, dIdt, dAdt, dRdt, dTdt, dHdt, dE_sdt, dDdt, dH_sdt]


# Initial conditions
S0 = 60000000
E0 = 100
I0 = 100
A0 = 100
R0 = 0
T0 = 0
H0 = 0
E_s0 = 0
D0 = 0
H_s0 = 0

# Parameters
beta = 0.25
gamma = 1 / 7
alpha = 0.2
mu = 0.2
eta = 0.2
rho = 0.1
theta = 0.2
delta = rho / (1 - rho)

# Time vector
t = np.linspace(0, 300, 300)

# Initial condition vector
y0 = [S0, E0, I0, A0, R0, T0, H0, E_s0, D0, H_s0]

# Integrate the SIDARTHE equations over the time grid
sol = odeint(sidarthe, y0, t, args=(beta, gamma, delta, alpha, mu, eta, rho, theta))

# Plotting
import matplotlib.pyplot as plt

plt.plot(t, sol[:, 0], 'b', label='S(t)')
plt.plot(t, sol[:, 1], 'y', label='E(t)')
plt.plot(t, sol[:, 2], 'r', label='I(t)')
plt.plot(t, sol[:, 3], 'g', label='A(t)')
plt.plot(t, sol[:, 4], 'c', label='R(t)')
plt.plot(t, sol[:, 5], 'm', label='T(t)')
plt.plot(t, sol[:, 6], 'k', label='H(t)')
plt.plot(t, sol[:, 7], 'tab:orange', label='E_s(t)')
plt.plot(t, sol[:, 8], 'tab:brown', label='D(t)')
plt.plot(t, sol[:, 9], 'tab:pink', label='H_s(t)')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend(loc='best')
plt.grid()
plt.show()
