import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, population, initial_infected, duration):
    N = population
    I0 = initial_infected
    S0 = N - I0
    R0 = 0
    t = np.linspace(0, duration, duration+1)
    S = np.zeros(duration+1)
    I = np.zeros(duration+1)
    R = np.zeros(duration+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = S0, I0, R0

    for i in range(duration):
        y = S[i], I[i], R[i]
        tspan = t[i], t[i+1]
        S[i+1], I[i+1], R[i+1] = rk4_step(deriv, y, tspan, N, beta, gamma)

    return t, S, I, R


def rk4_step(deriv, y, tspan, N, beta, gamma):
    t0, t1 = tspan
    dt = t1 - t0
    dt2 = dt / 2.0
    dt6 = dt / 6.0
    k1 = deriv(y, t0, N, beta, gamma)
    k2 = deriv(y + dt2 * k1, t0 + dt2, N, beta, gamma)
    k3 = deriv(y + dt2 * k2, t0 + dt2, N, beta, gamma)
    k4 = deriv(y + dt * k3, t1, N, beta, gamma)
    y_next = y + dt6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next


# Example usage
beta = 0.2
gamma = 0.1
population = 1000
initial_infected = 1
duration = 100

t, S, I, R = SIR_model(beta, gamma, population, initial_infected, duration)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
