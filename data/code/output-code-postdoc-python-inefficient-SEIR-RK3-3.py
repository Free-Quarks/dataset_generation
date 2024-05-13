import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, I0, E0, R0, t_max):
    def seir_deriv(y, t, beta, sigma, gamma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, t_max, t_max + 1)
    y0 = N - I0 - E0 - R0, E0, I0, R0
    ret = odeint(seir_deriv, y0, t, args=(beta, sigma, gamma, N))
    S, E, I, R = ret.T

    return S, E, I, R


# Example usage
beta = 0.2
sigma = 1 / 5.2
gamma = 1 / 2.3
N = 1000
I0, E0, R0 = 1, 0, 0
t_max = 160

S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered/Removed')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SEIR Model')
plt.show()
