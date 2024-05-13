import numpy as np
import matplotlib.pyplot as plt


def serid_rk3(S0, I0, R0, N, beta, gamma, delta, t_max, dt):
    def f(t, y):
        S, I, R = y
        dS = -beta*S*I/N
        dI = beta*S*I/N - (gamma+delta)*I
        dR = gamma*I
        return np.array([dS, dI, dR])
    
    t = np.arange(0, t_max, dt)
    y = np.zeros((len(t), 3))
    y[0] = np.array([S0, I0, R0])
    
    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + dt/2, y[i-1] + dt/2 * k1)
        k3 = f(t[i-1] + dt, y[i-1] - dt*k1 + 2*dt*k2)
        y[i] = y[i-1] + dt/6 * (k1 + 4*k2 + k3)
    
    return t, y[:, 0], y[:, 1], y[:, 2]


S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0
beta = 0.2
gamma = 0.1
delta = 0.05


t_max = 100
dt = 0.1


t, S, I, R = serid_rk3(S0, I0, R0, N, beta, gamma, delta, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using RK3')
plt.legend()
plt.show()

