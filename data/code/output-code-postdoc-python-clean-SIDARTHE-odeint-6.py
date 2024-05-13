import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe(y, t, beta, delta, rho, alpha, theta, epsilon):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta*S*(I + alpha*A + epsilon*E)/N
    dIdt = beta*S*(I + alpha*A + epsilon*E)/N - delta*I
    dDdt = (1-rho)*delta*I
    dAdt = rho*delta*I - theta*A
    dRdt = (1-alpha)*delta*I + (1-theta)*A
    dTdt = theta*A
    dHdt = epsilon*E
    dEdt = beta*S*(I + alpha*A + epsilon*E)/N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]

y0 = [999999, 1, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 30, 1000)
beta = 0.2
alpha = 0.5
delta = 0.1
rho = 0.2
theta = 0.1
epsilon = 0.01

result = odeint(sidarthe, y0, t, args=(beta, delta, rho, alpha, theta, epsilon))

plt.plot(t, result[:, 0], label='S')
plt.plot(t, result[:, 1], label='I')
plt.plot(t, result[:, 2], label='D')
plt.plot(t, result[:, 3], label='A')
plt.plot(t, result[:, 4], label='R')
plt.plot(t, result[:, 5], label='T')
plt.plot(t, result[:, 6], label='H')
plt.plot(t, result[:, 7], label='E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.show()
