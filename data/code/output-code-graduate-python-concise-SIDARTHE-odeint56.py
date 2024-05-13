import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function implementing the SIDARTHE model

def sidarthe_model(y, t, alpha, beta, sigma, gamma, delta1, delta2, delta3, delta4, epsilon1, epsilon2, epsilon3, epsilon4):
    S, I, D, A, R, T, H, E = y

    # Differential equations
    dSdt = -alpha * S * (I + A + R + T + H + E)
    dIdt = alpha * S * (I + A + R + T + H + E) - beta * I
    dDdt = delta1 * sigma * beta * I - gamma * D
    dAdt = (1 - delta1) * sigma * beta * I - delta2 * gamma * A
    dRdt = (1 - delta3) * (1 - delta4) * gamma * (D + A) - epsilon1 * R
    dTdt = delta3 * (1 - delta4) * gamma * (D + A) - epsilon2 * T
    dHdt = epsilon1 * R + epsilon2 * T - epsilon3 * H
    dEdt = epsilon3 * H - epsilon4 * E

    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]

# Parameters
alpha = 0.35
beta = 0.2
sigma = 0.1
gamma = 0.1

# Initial conditions
y0 = [999, 1, 0, 0, 0, 0, 0, 0]

# Time points
t = np.linspace(0, 100, 100)

# Solve the differential equations
result = odeint(sidarthe_model, y0, t, args=(alpha, beta, sigma, gamma, delta1, delta2, delta3, delta4, epsilon1, epsilon2, epsilon3, epsilon4))

# Plot the results
plt.plot(t, result[:, 0], label='S')
plt.plot(t, result[:, 1], label='I')
plt.plot(t, result[:, 2], label='D')
plt.plot(t, result[:, 3], label='A')
plt.plot(t, result[:, 4], label='R')
plt.plot(t, result[:, 5], label='T')
plt.plot(t, result[:, 6], label='H')
plt.plot(t, result[:, 7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
