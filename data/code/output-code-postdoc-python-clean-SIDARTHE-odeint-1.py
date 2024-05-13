import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that returns dS/dt, dI/dt, dR/dt, dA/dt, dT/dt, dH/dt, dE/dt

def sidarthe(y, t, N, beta, gamma, alpha, rho, theta, delta, mu):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - (1 - rho) * gamma * I - rho * theta * I
    dDdt = delta * theta * I
    dAdt = (1 - rho) * gamma * I - alpha * mu * A
    dRdt = rho * theta * I + alpha * mu * A
    dTdt = delta * theta * I
    dHdt = (1 - delta) * theta * I
    dEdt = beta * S * (I + alpha * A) / N
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


# Total population, N.
N = 100000
# Initial number of infected, recovered and dead individuals, I0, R0 and D0.
I0, D0, A0, R0, T0, H0, E0 = 1, 0, 0, 0, 0, 0, N-1
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - D0

# Contact rate, beta, and mean recovery rate, gamma, and mean death rate, alpha.
beta, gamma, alpha = 0.2, 1./10, 1./10
# percentage of asymptomatic individuals
rho = 0.6
# death rate of symptomatic individuals
theta = 0.02
# fraction of asymptomatic to symptomatic individuals
delta = 0.1
# fraction of asymptomatic individuals that become severe cases
mu = 0.2

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, D0, A0, R0, T0, H0, E0

# Integrate the SIDARTHE equations over the time grid, t.
solution = odeint(sidarthe, y0, t, args=(N, beta, gamma, alpha, rho, theta, delta, mu))
S, I, D, A, R, T, H, E = solution.T

# Plot the data on three separate curves for S(t), I(t), D(t), A(t), R(t), H(t), E(t)
plt.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, D/1000, 'g', alpha=0.5, lw=2, label='Dead')
plt.plot(t, A/1000, 'y', alpha=0.5, lw=2, label='Asymptomatic')
plt.plot(t, R/1000, 'm', alpha=0.5, lw=2, label='Recovered')
plt.plot(t, H/1000, 'c', alpha=0.5, lw=2, label='Hospitalized')
plt.plot(t, E/1000, 'k', alpha=0.5, lw=2, label='Exposed')
plt.xlabel('Time (days)')
plt.ylabel('Number (thousands)')
plt.title('SIDARTHE Model')
plt.legend()
plt.grid(True)
plt.show()
