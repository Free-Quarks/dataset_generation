import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe(y, t, beta, gamma, delta, alpha, rho, mu):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + delta + alpha) * I
    dDdt = delta * I - (rho + mu) * D
    dAdt = alpha * I - rho * A
    dRdt = gamma * I + rho * (A + D)
    dTdt = mu * D
    dHdt = rho * A
    dEdt = beta * S * I / N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, rho, mu, days):
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    t = np.linspace(0, days, days)
    result = odeint(sidarthe, y0, t, args=(beta, gamma, delta, alpha, rho, mu))
    return result[:, 1], result[:, 2], result[:, 3], result[:, 4], result[:, 5], result[:, 6], result[:, 7], result[:, 0]


# Example usage
N = 1000000
I0 = 10
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0
beta = 0.2
gamma = 0.1
alpha = 0.05
delta = 0.01
rho = 0.01
mu = 0.01
days = 100

S, I, D, A, R, T, H, E = sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, rho, mu, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Critical')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
