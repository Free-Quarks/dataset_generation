import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, t_end, dt):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_end, int(t_end/dt))

    S = N - I0 - R0
    y0 = S, I0, R0
    y = np.zeros((len(t), 3))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = dt * np.array(deriv(y[i-1], t[i-1], N, beta, gamma))
        k2 = dt * np.array(deriv(y[i-1] + 0.5*k1, t[i-1] + 0.5*dt, N, beta, gamma))
        k3 = dt * np.array(deriv(y[i-1] - k1 + 2*k2, t[i-1] + dt, N, beta, gamma))
        y[i] = y[i-1] + (1/6) * (k1 + 4*k2 + k3)

    S, I, R = y.T

    return t, S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0, R0 = 1, 0
t_end = 100
dt = 0.1


t, S, I, R = sir_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
