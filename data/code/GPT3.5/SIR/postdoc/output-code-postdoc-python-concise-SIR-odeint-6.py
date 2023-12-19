import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_SIR_model(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, I0, R0
    result = odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S, I, R = result.T
    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
days = 160
S, I, R = simulate_SIR_model(N, I0, R0, beta, gamma, days)

plt.plot(S, 'b', label='Susceptible')
plt.plot(I, 'r', label='Infected')
plt.plot(R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
