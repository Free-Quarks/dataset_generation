from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def SIDARTHE(y, t, beta, gamma, delta, alpha, rho, epsilon):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (gamma * I) - (delta * I) - (alpha * I)
    dDdt = delta * I
    dAdt = alpha * I - (rho * A) - (epsilon * A)
    dRdt = gamma * I + rho * A
    dTdt = epsilon * A
    dHdt = delta * I
    dEdt = rho * A
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


y0 = [999999, 1, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 200, 1000)
beta = 0.2
gamma = 0.1
delta = 0.01
alpha = 0.03
rho = 0.01
epsilon = 0.05
sol = odeint(SIDARTHE, y0, t, args=(beta, gamma, delta, alpha, rho, epsilon))

plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='D')
plt.plot(t, sol[:, 3], label='A')
plt.plot(t, sol[:, 4], label='R')
plt.plot(t, sol[:, 5], label='T')
plt.plot(t, sol[:, 6], label='H')
plt.plot(t, sol[:, 7], label='E')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
