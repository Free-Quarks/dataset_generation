import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe_model(y, t, alpha, dt, rho, theta, epsilon, mu): 
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -alpha * S * (I + rho * A + dt * T) / N
    dIdt = alpha * S * (I + rho * A + dt * T) / N - theta * I - epsilon * I
    dDdt = epsilon * mu * I - mu * D
    dAdt = (1 - epsilon) * mu * I - rho * A
    dRdt = theta * I
    dTdt = dt * T
    dHdt = rho * A
    dEdt = alpha * S * (I + rho * A + dt * T) / N - epsilon * I
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]

# Initial conditions
S0 = 60000000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

# Parameters
alpha = 0.2
rho = 0.4
theta = 0.01
epsilon = 0.03
mu = 0.01

# Time points
t = np.linspace(0, 100, 100)

# Solve ODE
sol = odeint(sidarthe_model, [S0, I0, D0, A0, R0, T0, H0, E0], t, args=(alpha, rho, theta, epsilon, mu))

# Plot results
plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='D')
plt.plot(t, sol[:, 3], label='A')
plt.plot(t, sol[:, 4], label='R')
plt.plot(t, sol[:, 5], label='T')
plt.plot(t, sol[:, 6], label='H')
plt.plot(t, sol[:, 7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

