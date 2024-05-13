```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y, t, N, beta, sigma, gamma, delta, alpha, rho):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - (sigma + delta) * I
    dDdt = delta * I
    dAdt = alpha * A * (I + rho * T) / N - (gamma + sigma) * A
    dRdt = sigma * (I + A) - gamma * R
    dTdt = rho * A - gamma * T
    dHdt = delta * I
    dEdt = sigma * (I + A) - gamma * E
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def solver(N, beta, sigma, gamma, delta, alpha, rho, S0, I0, D0, A0, R0, T0, H0, E0, days):
    t = np.linspace(0, days, days)
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    ret = odeint(model, y0, t, args=(N, beta, sigma, gamma, delta, alpha, rho))
    S, I, D, A, R, T, H, E = ret.T
    return S, I, D, A, R, T, H, E


N = 100000
beta = 0.2
sigma = 1/5.2
gamma = 1/2.9
alpha = 0.6
rho = 0.2
delta = 0.006
S0, I0, D0, A0, R0, T0, H0, E0 = N-1, 1, 0, 0, 0, 0, 0, 0
days = 100
S, I, D, A, R, T, H, E = solver(N, beta, sigma, gamma, delta, alpha, rho, S0, I0, D0, A0, R0, T0, H0, E0, days)

plt.plot(S, 'b', label='Susceptible')
plt.plot(I, 'r', label='Infected')
plt.plot(D, 'g', label='Deaths')
plt.plot(A, 'c', label='Asymptomatic')
plt.plot(R, 'm', label='Recovered')
plt.plot(T, 'y', label='Tested')
plt.plot(H, 'k', label='Hospitalized')
plt.plot(E, 'y', label='Exposed')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SIDARTHE Model Simulation')
plt.legend()
plt.show()```
