import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 1/10

t = np.linspace(0, 160, 160)

y0 = S0, I0, R0

ret = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
