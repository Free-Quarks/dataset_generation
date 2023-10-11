import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that defines the SIDARTHE model

def sidarthe_model(y, t, alpha_i, alpha_d, alpha_t, beta, gamma_i, gamma_d, gamma_h, gamma_a, gamma_r, epsilon, delta, rho, mu):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -alpha_i * S * I / N - alpha_d * S * D / N - alpha_t * S * T / N
    dIdt = alpha_i * S * I / N - (gamma_i + gamma_d + gamma_h + epsilon) * I
    dDdt = alpha_d * S * D / N + (1 - delta) * epsilon * I - (gamma_d + rho) * D
    dAdt = alpha_t * S * T / N - (gamma_a + beta) * A
    dRdt = gamma_i * I + gamma_a * A - (gamma_r + mu) * R
    dTdt = delta * epsilon * I - mu * T
    dHdt = gamma_h * I + rho * D - mu * H
    dEdt = beta * A - mu * E
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]

# Function to solve the ODE system

def solve_sidarthe_model(y0, t, alpha_i, alpha_d, alpha_t, beta, gamma_i, gamma_d, gamma_h, gamma_a, gamma_r, epsilon, delta, rho, mu):
    return odeint(sidarthe_model, y0, t, args=(alpha_i, alpha_d, alpha_t, beta, gamma_i, gamma_d, gamma_h, gamma_a, gamma_r, epsilon, delta, rho, mu))

# Parameters

N = 100000
alpha_i = 0.35
alpha_d = 0.4
alpha_t = 0.25
beta = 0.1
epsilon = 0.1
delta = 0.1
rho = 0.05
mu = 0.05
gamma_i = 0.1
gamma_d = 0.1
gamma_h = 0.05
gamma_a = 0.1
gamma_r = 0.1

# Initial conditions

y0 = [99999, 1, 0, 0, 0, 0, 0, 0]

# Time vector

t = np.linspace(0, 100, 100)

# Solve the ODE system

solution = solve_sidarthe_model(y0, t, alpha_i, alpha_d, alpha_t, beta, gamma_i, gamma_d, gamma_h, gamma_a, gamma_r, epsilon, delta, rho, mu)

# Plot the results

plt.figure(figsize=(10,6))
plt.plot(t, solution)
plt.legend(['S', 'I', 'D', 'A', 'R', 'T', 'H', 'E'])
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.grid(True)
plt.show()
