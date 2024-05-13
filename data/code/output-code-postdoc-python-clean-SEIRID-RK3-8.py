import numpy as np
import matplotlib.pyplot as plt

def seirid_model(beta, sigma, gamma, rho, delta, N, I0, E0, R0, D0, T):
    # Total population
    S0 = N - I0 - E0 - R0 - D0
    # Initial conditions
    y0 = S0, E0, I0, R0, D0
    
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days)
    # A grid of time points (in days)
    t = np.linspace(0, T, T)
    
    # The SEIRID model differential equations.
    def deriv(y, t, N, beta, sigma, gamma, rho, delta):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E - rho * E
        dIdt = sigma * E - gamma * I - delta * I
        dRdt = gamma * I
        dDdt = rho * E + delta * I
        return dSdt, dEdt, dIdt, dRdt, dDdt
    
    # Integrate the SEIRID equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, rho, delta))
    S, E, I, R, D = ret.T
    
    return t, S, E, I, R, D

# Example usage
# Parameters
N = 1000
I0, E0, R0, D0 = 1, 0, 0, 0
beta, sigma, gamma, rho, delta = 0.2, 1/5, 1/10, 1/5, 1/7
T = 200

# Run simulation
t, S, E, I, R, D = seirid_model(beta, sigma, gamma, rho, delta, N, I0, E0, R0, D0, T)

# Plot results
fig, ax = plt.subplots()
ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_ylim(0, N)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(True)
ax.legend()
plt.show()
