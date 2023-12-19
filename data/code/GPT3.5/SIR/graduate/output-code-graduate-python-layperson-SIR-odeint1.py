import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    sol = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = sol.T
    return S, I, R


S0 = 990
I0 = 10
R0 = 0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, 100)
S, I, R = simulate_SIR_model(S0, I0, R0, beta, gamma, t)

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
