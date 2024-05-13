import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, R0, E0, T):
    def deriv(y, t, N, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = N - I0 - R0 - E0, E0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
    S, E, I, R = ret.T

    return t, S, E, I, R


N = 10000  # population size
beta = 0.2  # infection rate
sigma = 1 / 5  # incubation period
gamma = 1 / 14  # recovery rate
I0, R0, E0 = 1, 0, 0  # initial conditions
T = 100  # time span

t, S, E, I, R = seir_model(beta, sigma, gamma, N, I0, R0, E0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

