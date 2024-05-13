import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sidarthe_model(t, y, alpha, beta, gamma, delta, epsilon, zeta, eta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dsdt = -alpha * S * (I + delta * A + epsilon * T) / N
    didt = alpha * S * (I + delta * A + epsilon * T) / N - (beta + gamma) * I
    dddt = gamma * I
    dadt = beta * I - (zeta + eta) * A
    drdt = zeta * A
    dtdt = eta * A
    dhdt = delta * alpha * S * A / N
    dedt = epsilon * alpha * S * T / N
    return [dsdt, didt, dddt, dadt, drdt, dtdt, dhdt, dedt]


def simulate_sidarthe_model(S, I, D, A, R, T, H, E, alpha, beta, gamma, delta, epsilon, zeta, eta, t_start, t_end, num_points):
    y0 = [S, I, D, A, R, T, H, E]
    t = np.linspace(t_start, t_end, num_points)
    solution = solve_ivp(sidarthe_model, (t_start, t_end), y0, t_eval=t, args=(alpha, beta, gamma, delta, epsilon, zeta, eta))
    return solution.t, solution.y


# Example usage
S = 10000
I = 100
D = 50
A = 200
R = 500
T = 10
H = 0
E = 100
alpha = 0.2
beta = 0.1
gamma = 0.05
delta = 0.1
epsilon = 0.15
zeta = 0.01
eta = 0.02
t_start = 0
t_end = 100
dt = 0.1

# Simulate the SIDARTHE model
t, y = simulate_sidarthe_model(S, I, D, A, R, T, H, E, alpha, beta, gamma, delta, epsilon, zeta, eta, t_start, t_end, int((t_end-t_start)/dt)+1)

# Plot the results
plt.plot(t, y[0], label='S')
plt.plot(t, y[1], label='I')
plt.plot(t, y[2], label='D')
plt.plot(t, y[3], label='A')
plt.plot(t, y[4], label='R')
plt.plot(t, y[5], label='T')
plt.plot(t, y[6], label='H')
plt.plot(t, y[7], label='E')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
