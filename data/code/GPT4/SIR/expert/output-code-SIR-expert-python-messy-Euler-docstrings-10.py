from numpy import linspace, zeros
from matplotlib import pyplot as plt

# Initial population parameter
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 

# A grid of time points (in days)
t = linspace(0, 160, 160)

def deriv_SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
S, I, R = zeros(len(t)), zeros(len(t)), zeros(len(t))
S[0], I[0], R[0] = y0

dt = t[1] - t[0]

for i in range(1, len(t)):
    dSdt, dIdt, dRdt = deriv_SIR((S[i-1], I[i-1], R[i-1]), t[i-1], N, beta, gamma)
    S[i] = S[i-1] + dSdt * dt
    I[i] = I[i-1] + dIdt * dt
    R[i] = R[i-1] + dRdt * dt

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.legend()
plt.show()

