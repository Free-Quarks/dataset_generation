from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
def __model__(y, t, N, b, k):
    S, I, R = y
    dSdt = -b * S * I / N
    dIdt = b * S * I / N - k * I
    dRdt = k * I
    return dSdt, dIdt, dRdt
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
b, k = 0.2, 1./10 
t = np.linspace(0, 160, 160)
y0 = S0, I0, R0
ret = odeint(__model__, y0, t, args=(N, b, k))
S, I, R = ret.T
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
plt.show()
