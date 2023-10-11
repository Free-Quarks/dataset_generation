import numpy as np
import matplotlib.pyplot as plt


def serid_model(beta, gamma, delta, N, I0, R0, D0, T):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - D0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # Death rate, delta (in 1/days).
    # A grid of time points (in days)
    t = np.linspace(0, T, T+1)
    # Initial conditions vector
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    def deriv(y, t, N, beta, gamma, delta):
        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - (gamma + delta) * I
        dRdt = gamma * I
        dDdt = delta * I
        return dSdt, dIdt, dRdt, dDdt
    
    return t, odeint(deriv, y0, t, args=(N, beta, gamma, delta))


# Example usage:
T = 100  # time steps
N = 1000  # total population
I0, R0, D0 = 1, 0, 0  # initial infected, recovered, deaths
beta, gamma, delta = 0.2, 1/10, 1/20  # infection rate, recovery rate, death rate

# Run the SERID model
t, y = serid_model(beta, gamma, delta, N, I0, R0, D0, T)

# Plot the results
plt.plot(t, y[:, 0], 'b', label='Susceptible')
plt.plot(t, y[:, 1], 'r', label='Infected')
plt.plot(t, y[:, 2], 'g', label='Recovered')
plt.plot(t, y[:, 3], 'k', label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SERID Model')
plt.legend()
plt.grid(True)
plt.show()

